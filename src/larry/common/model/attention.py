import math

import torch
import torch.nn.functional as F
from rotary_embedding_torch import RotaryEmbedding
from rotary_embedding_torch import apply_rotary_emb
from torch import nn

from larry.common.config.model.model_configs import AttentionModuleConfig, FSMNAttentionConfig
from larry.common.model.kv_cache import KVCache
from larry.common.optim.qk_clip import QKModule


class LarryAttention(nn.Module, QKModule):
    def __init__(self, config: AttentionModuleConfig):
        super().__init__()
        self.config = config

        self.n_query_groups = config.n_query_groups
        self.d_queries = config.d_model // config.n_heads
        self.d_values = config.d_model // config.n_heads // self.n_query_groups
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
        self.rotary_global = self.rotary_local = None
        if config.positional_encoding_name == "rotary":
            self.rotary_embedding = RotaryEmbedding(dim=config.rotary_embedding_dim)
            if self.config.use_split_rope:
                self._split_rope_half = config.rotary_embedding_dim // 2
                self.rotary_global = RotaryEmbedding(dim=self._split_rope_half)
                self.rotary_local = RotaryEmbedding(dim=self._split_rope_half)

        self.o_proj = nn.Linear(self.n_heads * self.d_values, config.d_model)

        self.attn_dropout = nn.Dropout(config.attention_dropout_p)
        self.proj_dropout = nn.Dropout(config.proj_dropout_p)

        self.is_causal = config.is_causal
        max_positions = config.max_position_embeddings

        if self.is_causal:
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
        if self.config.use_split_rope and self.rotary_global and self.rotary_local and position_ids is not None:
            # (N, t, 2) = [:, :, 0] global, [:, :, 1] local (local may be fractional)
            global_pids = position_ids[..., 0].to(queries.device).float()
            local_pids = position_ids[..., 1].to(queries.device).float()

            # RotaryEmbedding accepts batched positions ((N, t) -> (N, t, dim)) and matches
            # the per-row result exactly, so this must not be a Python loop: it runs once
            # per attention call, i.e. ~n_blocks * n_iterations times per forward.
            fg = self.rotary_global(global_pids).unsqueeze(1)  # (N, 1, t, half)
            fl = self.rotary_local(local_pids).unsqueeze(1)

            queries = apply_rotary_emb(fg, queries, start_index=0)
            keys = apply_rotary_emb(fg, keys, start_index=0)
            queries = apply_rotary_emb(fl, queries, start_index=self._split_rope_half)
            keys = apply_rotary_emb(fl, keys, start_index=self._split_rope_half)
        elif self.rotary_embedding:
            # rotate_queries_or_keys supports offset parameter for cached generation
            queries = self.rotary_embedding.rotate_queries_or_keys(queries, offset=position_offset)
            keys = self.rotary_embedding.rotate_queries_or_keys(keys, offset=position_offset)
        return queries, keys

    def _kv_cache(self, kv_cache: KVCache, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is not None and kv_cache.key_cache is not None and kv_cache.value_cache is not None:
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

            k = torch.cat([kv_cache.key_cache, k], dim=2)
            v = torch.cat([kv_cache.value_cache, v], dim=2)
        else:
            # expand k/v for gqa
            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)
        return k, v

    def _sliced_causal_mask(self, q_seq_len: int, k_seq_len: int) -> torch.Tensor:
        # For cached generation with Huginn-style cache, position_offset tracks the global position while k_seq_len
        # may differ across cache slots
        # use k_seq_len-based offset so the mask matches the actual key dimension
        eff_offset = max(k_seq_len - q_seq_len, 0)  # query positions relative to key positions
        required_size = max(eff_offset + q_seq_len, k_seq_len)

        if required_size > self.causal_mask.shape[-1]:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError(
                    f"Causal mask buffer too small: required_size={required_size} > "
                    f"self.causal_mask.shape[-1]={self.causal_mask.shape[-1]}. "
                    f"Increase MegaTransformerBlockConfig.max_position_embeddings to "
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
            causal_mask: torch.Tensor | None,
            active_mask: torch.Tensor | None
    ) -> torch.Tensor:
        attention_scores = torch.matmul(queries, keys.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.d_queries)

        if causal_mask is not None:
            attention_scores = attention_scores.masked_fill(causal_mask == 0, float("-inf"))

        if active_mask is not None:
            attention_scores = attention_scores.masked_fill(active_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))
        return attention_scores

    def _qk_max_logit(
            self, q: torch.Tensor,
            k: torch.Tensor,
            causal_mask: torch.Tensor | None,
            active_mask: torch.Tensor | None
    ) -> None:
        with torch.no_grad():
            attention_scores = self._masked_attention(q.float(), k.float(), causal_mask, active_mask)

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
    ) -> tuple[torch.Tensor, KVCache | None]:
        is_cross_attention = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross_attention else hidden_states

        batch_size, _ = hidden_states.shape[:2]

        queries: torch.Tensor = self.q_proj(hidden_states)
        keys: torch.Tensor = self.k_proj(kv_input)
        values: torch.Tensor = self.v_proj(kv_input)

        # reshape to head format before applying rope
        # Q: (N, seq, n_heads * d_queries) -> (N, n_heads, seq, d_queries)
        # K: (N, seq, n_query_groups * d_queries) -> (N, n_query_groups, seq, d_queries)
        # V: (N, seq, n_query_groups * d_values) -> (N, n_query_groups, seq, d_values)
        queries = queries.view(batch_size, -1, self.n_heads, self.d_queries).permute(0, 2, 1, 3).contiguous()
        keys = keys.view(batch_size, -1, self.n_query_groups, self.d_queries).permute(0, 2, 1, 3).contiguous()
        values = values.view(batch_size, -1, self.n_query_groups, self.d_values).permute(0, 2, 1, 3).contiguous()

        # apply rope with position offset for correct positions during generation
        if self.rotary_embedding is not None:
            queries, keys = self._apply_rotary_embedding(queries, keys, position_ids, position_offset)

        if kv_cache:
            keys, values = self._kv_cache(kv_cache, keys, values)

        new_kv_cache = None
        if use_cache:
            new_kv_cache = KVCache()
            new_kv_cache.key_cache = keys.clone()
            new_kv_cache.value_cache = values.clone()

        q_seq_len = queries.shape[2]
        k_seq_len = keys.shape[2]

        causal_mask = None
        if self.is_causal and not is_cross_attention:
            causal_mask = self._sliced_causal_mask(q_seq_len, k_seq_len)

        active_mask = encoder_attention_mask if is_cross_attention else attention_mask

        if self.qk_probe_active:
            self._qk_max_logit(queries, keys, causal_mask, active_mask)

        attn_mask = None
        if causal_mask is not None and not (active_mask is None and q_seq_len == k_seq_len):
            attn_mask = causal_mask != 0

        if active_mask is not None:
            pad_bool = active_mask.unsqueeze(1).unsqueeze(2) != 0
            attn_mask = pad_bool if attn_mask is None else (attn_mask & pad_bool)

        # explicitly let each token attend to itself to avoid infs/NaNs
        if attn_mask is not None:
            eye = torch.eye(
                q_seq_len,
                k_seq_len,
                dtype=torch.bool,
                device=attn_mask.device
            )[None, None, -q_seq_len:]
            attn_mask = attn_mask | eye

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

        return output, new_kv_cache


class FSMNAttention(LarryAttention):
    """Self-attention plus an FSMN memory branch (SAN-M): a depthwise conv over time, added to the attention output."""

    def __init__(self, config: FSMNAttentionConfig) -> None:
        super().__init__(config)
        d_model = config.d_model
        kernel_size = config.fsmn_kernel_size

        self.memory_proj = nn.Linear(d_model, d_model, bias=config.use_qkv_bias)
        self.fsmn = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)

        left = (kernel_size - 1) // 2
        self.fsmn_padding = (left, kernel_size - 1 - left)
        self.memory_dropout = nn.Dropout(config.proj_dropout_p)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        attended, _ = super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        memory = self._memory(hidden_states, attention_mask)
        return attended + memory, None

    def _memory(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        memory_in = self.memory_proj(hidden_states)  # [B, T, D]

        # the conv mixes neighboring frames and ignores the attention mask, so padded frames
        # must be zeroed before it (they hold the projection's bias, not zero) and after it
        keep = None
        if attention_mask is not None:
            keep = attention_mask.unsqueeze(-1).to(memory_in.dtype)  # [B, T, 1], 1 on real frames
            memory_in = memory_in * keep

        x = F.pad(memory_in.transpose(1, 2), self.fsmn_padding)  # [B, D, T + k - 1]
        memory = self.fsmn(x).transpose(1, 2) + memory_in

        if keep is not None:
            memory = memory * keep

        return self.memory_dropout(memory)
