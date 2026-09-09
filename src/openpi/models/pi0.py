import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)


        # ── B1：VLM 尾部的动态槽位（dyn_slots=0 时下面什么都不建，等于上游）──────
        self.dyn_slots = config.dyn_slots
        self.dyn_ce_weight = config.dyn_ce_weight
        self.dyn_branch_sizes = tuple(config.dyn_branch_sizes)
        if self.dyn_slots > 0:
            assert sum(self.dyn_branch_sizes) == self.dyn_slots, (
                "dyn_branch_sizes 之和必须等于 dyn_slots"
            )
            # VLM 的宽度直接取 paligemma_config（gemma_2b=2048；dummy=64）。
            # 不能猜 —— 猜 2048 会在 dummy 变体上拼接失败。
            vlm_width = paligemma_config.width
            # 每个槽位一个可学习嵌入向量，作为 VLM 的额外输入 token
            self.dyn_slot_emb = nnx.Param(
                jax.random.normal(rngs.params(), (self.dyn_slots, vlm_width)) * 0.02
            )
            n_cls = config.dyn_codebook_size
            # ⚠️ 这里**必须**用字符串键的容器，不能用 list。
            # openpi 到处都在做 `flax.traverse_util.flatten_dict(params, sep="/")`
            # 把参数树拍平成 "a/b/c" —— 权重加载（weight_loaders.py:87）、orbax 存点、
            # FSDP 分片规则全靠它。list 会让路径里出现整数下标，`"/".join` 直接抛
            # `TypeError: sequence item 1: expected str instance, int found`。
            # 这条路径 forward/loss 的冒烟测试不走，所以 2026-09-09 第一次真起训练才炸出来。
            if config.dyn_ce_head_mode == "per_slot":
                # 12 个独立头：槽位 i 用 W_i，各自适配该位置的分布
                self.dyn_head_names = tuple(f"slot{i:02d}" for i in range(self.dyn_slots))
            elif config.dyn_ce_head_mode == "per_branch":
                # 3 个头：同一分支的 4 个槽位共享一个 W（它们索引的本来就是同一张码表）
                self.dyn_head_names = (
                    ("left", "right", "env")
                    if len(self.dyn_branch_sizes) == 3
                    else tuple(f"branch{j}" for j in range(len(self.dyn_branch_sizes)))
                )
            else:
                raise ValueError(f"未知的 dyn_ce_head_mode: {config.dyn_ce_head_mode}")
            self.dyn_heads = nnx.Dict(
                {name: nnx.Linear(vlm_width, n_cls, rngs=rngs) for name in self.dyn_head_names}
            )
            self.dyn_ce_head_mode = config.dyn_ce_head_mode
        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        # ── B1：在 image/text 之后追加 dyn_slots 个可学习槽位 ──────────────────
        # ar_mask 给 [True] + [False]*(n-1)：槽位自成一块（cumsum=1），于是
        #   · 槽位能 attend 到 image/text（cumsum 0 ≤ 1）✓ 它需要看图才能预测动态
        #   · image/text attend 不到槽位（0 < 1）✓ 用户选的"槽位只读"，前向计算不受污染
        # 动作专家的屏蔽做不到这里 —— make_attn_mask 是单调 cumsum，后面的块必然看得到
        # 前面的块。所以那一步在 compute_loss 里显式改掩码。
        if self.dyn_slots > 0:
            b = tokens[0].shape[0] if isinstance(tokens, list) else tokens.shape[0]
            slot_tokens = jnp.broadcast_to(
                self.dyn_slot_emb.value[None], (b, self.dyn_slots, self.dyn_slot_emb.value.shape[-1])
            )
            tokens.append(slot_tokens)
            input_mask.append(jnp.ones((b, self.dyn_slots), dtype=jnp.bool_))
            ar_mask += [True] + [False] * (self.dyn_slots - 1)

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1

        if self.dyn_slots > 0:
            n_prefix = prefix_tokens.shape[1]
            slot_lo, slot_hi = n_prefix - self.dyn_slots, n_prefix

            # (a) B1：动作专家看不到槽位。make_attn_mask 是单调 cumsum，后面的块必然
            #     看得到前面的块，光靠 ar_mask 屏蔽不掉，只能在这里把
            #     「suffix 行 × 槽位列」显式置 False。**这一行就是 B1 与 B2 的全部区别。**
            attn_mask = attn_mask.at[:, n_prefix:, slot_lo:slot_hi].set(False)

            # (b) 槽位不占位置编号。否则动作 token 的 RoPE 位置会整体后移 dyn_slots，
            #     与 B0 不可比（动作到图像的相对距离被拉远）。让动作 token 的位置从
            #     "image/text 之后"接着数，和 B0 完全一致；槽位与动作 token 位置编号
            #     重叠是安全的 —— 它们互不可见。
            base = jnp.cumsum(input_mask[:, :slot_lo], axis=1) - 1        # image/text: 0..n-1
            n_img_txt = base[:, -1:] + 1
            slot_pos = n_img_txt + jnp.arange(self.dyn_slots)[None, :]
            suffix_pos = n_img_txt + jnp.arange(suffix_tokens.shape[1])[None, :]
            positions = jnp.concatenate([base, slot_pos, suffix_pos], axis=1)

        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)              # [b, ah]

        # (c) B1 的辅助损失：12 路 CE。标签是师兄 tokenizer 离线算好的动态码。
        #     专家看不到槽位，所以这条损失影响模型的**唯一**路径是梯度回传去塑造
        #     共享的 VLM 权重 —— 这正是要回答的问题。
        if self.dyn_slots > 0 and observation.dyn_codes is not None:
            slot_out = prefix_out[:, -self.dyn_slots :]                   # [b, n_slot, width]
            logits = []
            if self.dyn_ce_head_mode == "per_slot":
                for i, name in enumerate(self.dyn_head_names):
                    logits.append(self.dyn_heads[name](slot_out[:, i]))
            else:  # per_branch：同一分支的槽位共用一个头
                i = 0
                for name, n in zip(self.dyn_head_names, self.dyn_branch_sizes, strict=True):
                    h = self.dyn_heads[name]
                    for _ in range(n):
                        logits.append(h(slot_out[:, i])); i += 1
            logits = jnp.stack(logits, axis=1)                            # [b, n_slot, n_cls]
            logp = jax.nn.log_softmax(logits, axis=-1)
            tgt = jax.nn.one_hot(observation.dyn_codes, logits.shape[-1])
            ce = -jnp.sum(logp * tgt, axis=-1)                            # [b, n_slot]
            if observation.dyn_codes_mask is not None:
                # t+k 越过 episode 末尾的样本没有合法标签，不参与 CE
                w = observation.dyn_codes_mask.astype(ce.dtype)[:, None]
                ce = ce * w
                denom = jnp.maximum(jnp.sum(w), 1.0) * ce.shape[1]
            else:
                denom = ce.size
            ce_mean = jnp.sum(ce) / denom
            # 摊回 [b, ah] 的形状，好让上层的 jnp.mean 得到 flow + w·CE
            flow_loss = flow_loss + self.dyn_ce_weight * ce_mean

        return flow_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
