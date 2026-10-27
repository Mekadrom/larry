import dataclasses
import math

import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb
from torch import nn

from larry.common.config.model.model_configs import AttentionModuleConfig, FSMNAttentionConfig
from larry.common.model.kv_cache import KVCache
from larry.common.model.model import LarryModelOutput
from larry.common.training.qk_clip import QKModule


@dataclasses.dataclass(kw_only=True)
class LarryAttentionOutput[KV](LarryModelOutput):
    kv_cache: KV | None = None


class LarryAttention(nn.Module, QKModule):
    """Generic attention module. Supports KV cache."""

    @staticmethod
    def make_rotary(dim: int) -> RotaryEmbedding:
        rotary = RotaryEmbedding(dim=dim)
        length = rotary.cache_max_seq_len
        with torch.no_grad():
            rotary.forward(torch.arange(length), seq_len=length)

        return rotary

    def __init__(self, config: AttentionModuleConfig) -> None:
        """Initializes all modules required for the provided configuration."""
        super().__init__()
        self.config = config

        self.n_query_groups = config.n_query_groups
        self.d_queries = config.d_model // config.n_heads
        self.d_values = config.d_model // config.n_heads
        self.n_heads = config.n_heads

        # for gqa, queries have full n_heads, keys/values have fewer n_query_groups (shared across query heads)
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.d_queries, bias=config.use_qkv_bias)
        self.k_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_queries, bias=config.use_qkv_bias)
        self.v_proj = nn.Linear(config.d_model, self.n_query_groups * self.d_values, bias=config.use_qkv_bias)

        # number of times to repeat k/v heads to match q heads
        if self.n_heads % self.n_query_groups != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_query_groups ({self.n_query_groups})")

        self.n_rep = self.n_heads // self.n_query_groups

        self.rotary_embedding = None
        self.rotary_global = self.rotary_local = self.rotary_local_2 = None
        if config.positional_encoding_name == "rotary":
            self.rotary_embedding = self.make_rotary(config.rotary_embedding_dim)
            if self.config.use_split_rope:
                self._split_rope_half = config.rotary_embedding_dim // 2
                self.rotary_global = self.make_rotary(self._split_rope_half)
                if config.split_rope_local_axes == 2:
                    if self._split_rope_half % 4 != 0 or self._split_rope_half < 8:
                        raise ValueError("two local RoPE axes need rotary_embedding_dim divisible by 8 and at least 16")

                    self._split_rope_quarter = self._split_rope_half // 2
                    self.rotary_local = self.make_rotary(self._split_rope_quarter)
                    self.rotary_local_2 = self.make_rotary(self._split_rope_quarter)
                else:
                    self.rotary_local = self.make_rotary(self._split_rope_half)

        self.o_proj = nn.Linear(self.n_heads * self.d_values, config.d_model)

        self.attn_dropout = nn.Dropout(config.attention_dropout_p)
        self.proj_dropout = nn.Dropout(config.proj_dropout_p)

        max_positions = config.max_seq_len

        if config.is_causal:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(max_positions, max_positions)).view(1, 1, max_positions, max_positions),
                persistent=False,
            )
        else:
            self.register_buffer("causal_mask", None)

    def _apply_rotary_embedding(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            position_ids: torch.Tensor | None = None,
            position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Given queries and keys, applies the configured RoPE manipulations.

        TODO: support 3-axis or n-axis m-rope (for image RoPE; global position dims, and row/column position dims)
        """
        if self.config.use_split_rope and self.rotary_global and self.rotary_local:
            n_axes = 1 + self.config.split_rope_local_axes
            if position_ids is None or position_ids.shape[-1] < n_axes:
                raise ValueError(f"split RoPE requires position_ids of shape (N, t, {n_axes}) or wider")

            # (N, t, >=2) = [:, :, 0] global, [:, :, 1] local (local may be fractional), [:, :, 2] second local axis
            global_pids = position_ids[..., 0].to(queries.device).float() + position_offset
            local_pids = position_ids[..., 1].to(queries.device).float()

            fg = self.rotary_global(global_pids).unsqueeze(1)  # (N, 1, t, half)
            fl = self.rotary_local(local_pids).unsqueeze(1)

            queries = apply_rotary_emb(fg, queries, start_index=0)
            keys = apply_rotary_emb(fg, keys, start_index=0)
            queries = apply_rotary_emb(fl, queries, start_index=self._split_rope_half)
            keys = apply_rotary_emb(fl, keys, start_index=self._split_rope_half)

            if self.rotary_local_2 is not None:
                local_2_pids = position_ids[..., 2].to(queries.device).float()
                fl2 = self.rotary_local_2(local_2_pids).unsqueeze(1)
                start = self._split_rope_half + self._split_rope_quarter
                queries = apply_rotary_emb(fl2, queries, start_index=start)
                keys = apply_rotary_emb(fl2, keys, start_index=start)
        elif self.rotary_embedding:
            # rotate_queries_or_keys supports offset parameter for cached generation
            queries = self.rotary_embedding.rotate_queries_or_keys(queries, offset=position_offset)
            keys = self.rotary_embedding.rotate_queries_or_keys(keys, offset=position_offset)

        return queries, keys

    def _kv_cache(
            self, kv_cache: KVCache | None, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """If the KV cache contains KV tensors, prepends those to the input sequence. Apply after rotary embedding."""
        if kv_cache is not None and kv_cache.key_cache is not None and kv_cache.value_cache is not None:
            k = torch.cat([kv_cache.key_cache, k], dim=2)
            v = torch.cat([kv_cache.value_cache, v], dim=2)
        return k, v

    def _expand_kv_groups(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeats each of the ``n_query_groups`` k/v heads so that every query head has a matching k/v head."""
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        return k, v

    def _sliced_causal_mask(self, q_seq_len: int, k_seq_len: int) -> torch.Tensor:
        """Produces a causal mask from the cached single large causal mask using input q and k sequence lengths."""
        eff_offset = max(k_seq_len - q_seq_len, 0)  # query positions relative to key positions
        required_size = max(eff_offset + q_seq_len, k_seq_len)

        if required_size > self.causal_mask.shape[-1]:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    f"Causal mask buffer too small: required_size={required_size} > "
                    f"self.causal_mask.shape[-1]={self.causal_mask.shape[-1]}. "
                    f"Increase max_seq_len to "
                    f"at least {required_size}. Or fix your batch collation."
                )

            new_size = max(required_size, 2 * self.causal_mask.shape[-1])
            self.causal_mask = torch.tril(
                torch.ones(new_size, new_size, device=self.causal_mask.device, dtype=self.causal_mask.dtype)
            ).view(1, 1, new_size, new_size)

        return self.causal_mask[:, :, eff_offset:eff_offset + q_seq_len, :k_seq_len]

    def _masked_attention(
            self,
            queries: torch.Tensor,
            keys: torch.Tensor,
            attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Performs manual masked attention. Does not use SDPA. Uses the same boolean mask that SDPA receives."""
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_queries)

        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(~attn_mask, float("-inf"))

        return attention_scores

    def _qk_max_logit(self, q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None) -> None:
        """Saves the maximum logit value from the attention scores for QK clipping."""
        with torch.no_grad():
            attention_scores = self._masked_attention(q.float(), k.float(), attn_mask)

            # (N, heads, q_seq_len, k_seq_len) -> per-head max over batch and both position axes.
            full_max = attention_scores.amax(dim=3).amax(dim=2).amax(dim=0)

            # A fully-masked row yields -inf; a head with no valid key contributes nothing.
            full_max = torch.where(torch.isfinite(full_max), full_max, torch.zeros_like(full_max))
            prev = self.qk_max_logit
            self.qk_max_logit = full_max if prev is None else torch.maximum(prev, full_max.to(prev.device))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            kv_cache: KVCache | None = None,
            position_offset: int = 0,
            use_cache: bool = False,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
    ) -> LarryAttentionOutput[KVCache]:
        """Main forward pass to multi-head attention.

        Supports GQA and causal or bidirectional attention. Defaults to using SDPA, only uses manual/materialized
        attention calculation for QK logit clipping when using muon. Supports KV caching, returns a simple tuple of the
        output logits/projected scores and the updated KVCache object.
        """
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states

        batch_size, _ = hidden_states.shape[:2]

        queries: torch.Tensor = self.q_proj(hidden_states)

        # reshape to head format before applying rope
        # Q: (N, seq, n_heads * d_queries) -> (N, n_heads, seq, d_queries)
        # K: (N, seq, n_query_groups * d_queries) -> (N, n_query_groups, seq, d_queries)
        # V: (N, seq, n_query_groups * d_values) -> (N, n_query_groups, seq, d_values)
        queries = queries.view(batch_size, -1, self.n_heads, self.d_queries).permute(0, 2, 1, 3).contiguous()

        cached_cross = (
                is_cross_attention
                and kv_cache is not None
                and kv_cache.key_cache is not None
                and kv_cache.value_cache is not None
        )
        if cached_cross:
            assert kv_cache is not None and kv_cache.key_cache is not None and kv_cache.value_cache is not None
            keys, values = kv_cache.key_cache, kv_cache.value_cache
        else:
            keys = self.k_proj(kv_input)
            values = self.v_proj(kv_input)
            keys = keys.view(batch_size, -1, self.n_query_groups, self.d_queries).permute(0, 2, 1, 3).contiguous()
            values = values.view(batch_size, -1, self.n_query_groups, self.d_values).permute(0, 2, 1, 3).contiguous()

        # apply rope with position offset for correct positions during generation
        # apply prior to KV cache prefixing using the position_offset to avoid recalculating rotary embedding for cached
        # values that already have this embedding applied
        if self.rotary_embedding is not None and not is_cross_attention:
            queries, keys = self._apply_rotary_embedding(queries, keys, position_ids, position_offset)

        if not is_cross_attention:
            keys, values = self._kv_cache(kv_cache, keys, values)

        new_kv_cache = None
        if use_cache:
            new_kv_cache = KVCache(key_cache=keys, value_cache=values)

        keys, values = self._expand_kv_groups(keys, values)

        q_seq_len = queries.shape[2]
        k_seq_len = keys.shape[2]

        causal_mask = None
        if self.config.is_causal and not is_cross_attention:
            causal_mask = self._sliced_causal_mask(q_seq_len, k_seq_len)

        active_mask = encoder_attention_mask if is_cross_attention else attention_mask

        attn_mask = None
        if active_mask is not None and active_mask.dim() == 4:
            attn_mask = active_mask != 0
        elif causal_mask is not None and not (active_mask is None and q_seq_len == k_seq_len):
            attn_mask = causal_mask != 0

        if active_mask is not None and active_mask.dim() != 4:
            pad_bool = active_mask.unsqueeze(1).unsqueeze(2) != 0
            attn_mask = pad_bool if attn_mask is None else attn_mask & pad_bool

        # explicitly let each token attend to itself to avoid infs/NaNs
        if attn_mask is not None and not is_cross_attention:
            query_positions = torch.arange(k_seq_len - q_seq_len, k_seq_len, device=attn_mask.device)
            key_positions = torch.arange(k_seq_len, device=attn_mask.device)
            eye = query_positions[:, None] == key_positions[None, :]
            attn_mask = attn_mask | eye[None, None]

        if self.qk_probe_active:
            probe_mask = attn_mask if attn_mask is not None or causal_mask is None else causal_mask != 0
            self._qk_max_logit(queries, keys, probe_mask)

        values = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=(attn_mask is None and causal_mask is not None),
            scale=1.0 / math.sqrt(self.d_queries),
        )
        values = values.permute(0, 2, 1, 3).contiguous()

        new_shape = values.size()[:-2] + (self.d_values * self.n_heads,)
        values = values.view(*new_shape)

        output = self.o_proj(values)
        output = self.proj_dropout(output)

        return LarryAttentionOutput(logits=output, kv_cache=new_kv_cache)


class FSMNAttention(LarryAttention):
    """Self-attention plus an FSMN memory branch (SAN-M): a depthwise conv over time, added to the attention output."""

    def __init__(self, config: FSMNAttentionConfig) -> None:
        """Sets up the additional conv memory modules."""
        super().__init__(config)
        d_model = config.d_model
        kernel_size = config.fsmn_kernel_size

        self.memory_proj = nn.Linear(d_model, d_model, bias=config.use_qkv_bias)
        self.fsmn = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)

        left = (kernel_size - 1) // 2
        self.fsmn_padding = (left, kernel_size - 1 - left)
        self.memory_dropout = nn.Dropout(config.proj_dropout_p)

    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
    ) -> LarryAttentionOutput:
        """Delegates to the super forward, applies the local convolutional "memory" module pass afterward.

        Applies convolutional local-context logits as residual. Does not support KV caching, and is bidirectional
        attention and cannot be overridden.
        """
        output = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        memory = self._memory(hidden_states, attention_mask)

        # apply memory as residual; doesn't support KV cache
        output.logits = output.logits + memory

        return output

    def _shifted_depthwise(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fsmn.weight[:, 0, :].to(x.dtype)
        kernel_size = weight.shape[-1]
        length = x.shape[-1] - kernel_size + 1
        out = x[..., :length] * weight[:, :1]
        for offset in range(1, kernel_size):
            out = out + x[..., offset:offset + length] * weight[:, offset:offset + 1]

        return out

    def _memory(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        """Applies a memory projection as a linear convolution."""
        memory_in = self.memory_proj(hidden_states)  # [B, T, D]

        # the conv mixes neighboring frames and ignores the attention mask, so padded frames
        # must be zeroed before it (they hold the projection's bias, not zero) and after it
        keep = None
        if attention_mask is not None:
            keep = attention_mask.unsqueeze(-1).to(memory_in.dtype)  # [B, T, 1], 1 on real frames
            memory_in = memory_in * keep

        x = F.pad(memory_in.transpose(1, 2), self.fsmn_padding)  # [B, D, T + k - 1]
        convolved = self._shifted_depthwise(x) if torch.compiler.is_compiling() else self.fsmn(x)
        memory = convolved.transpose(1, 2) + memory_in

        if keep is not None:
            memory = memory * keep

        return self.memory_dropout(memory)
