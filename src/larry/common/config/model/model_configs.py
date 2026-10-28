import dataclasses
from typing import Literal, Self

from larry.common.config.config import Config, TaggedConfig
from larry.common.utils.types import NormTypeEnum


@dataclasses.dataclass(kw_only=True)
class ModelConfig(TaggedConfig):
    """Superclass for configs for ``LarryModel`` implementations.

    Includes configuration for a maximum sequence length (since all ``LarryModel`` implementations work on sequential,
    varying length data) and a base d_model.
    """

    d_model: int = 16

    max_seq_len: int = 1024

    gradient_checkpointing: bool = False


@dataclasses.dataclass(kw_only=True)
class AttentionModuleConfig(ModelConfig):
    """Config class for all ``LarryModel`` implementations which require configuring ``LarryAttention`` submodules."""

    n_query_groups: int = 16
    n_heads: int = 16
    use_qkv_bias: bool = False

    positional_encoding_name: Literal["rotary"] = "rotary"

    use_split_rope: bool = False
    split_rope_local_axes: Literal[1, 2] = 1
    split_rope_scale: float = 7.5
    rotary_embedding_dim: int = 64

    attention_dropout_p: float = 0.1
    proj_dropout_p: float = 0.1

    is_causal: bool = True

    @classmethod
    def from_child(cls, config: AttentionModuleConfig) -> Self:
        """Instantiates an instance of ``AttentionModuleConfig``.

        Uses the applicable values from an instance of a subclass of ``AttentionModuleConfig`` to override defaults.
        """
        return cls(**{f.name: getattr(config, f.name) for f in dataclasses.fields(cls) if hasattr(config, f.name)})


@dataclasses.dataclass(kw_only=True)
class LLMBlockModuleConfig(AttentionModuleConfig):
    """Subclass implementation of ``AttentionModuleConfig``.

    For use in configuring block modules containing both instance(s) of an ``LarryAttention`` and an FFN.
    """

    pre_attn_norm_type: NormTypeEnum = "rms"
    pre_mlp_norm_type: NormTypeEnum = "rms"

    mlp_dropout_p: float = 0.1


@dataclasses.dataclass(kw_only=True)
class FSMNAttentionConfig(AttentionModuleConfig):
    """Subclass implementation of ``AttentionModuleConfig``.

    For use in configuring the specialized implementation of ``LarryAttention``, ``FSMNAttention`` which incorporates a
    CNN FSMN memory branch, and overrides the default value for ``is_causal`` to False for bidirectional attention.
    """

    fsmn_kernel_size: int = 31
    is_causal: bool = False


@dataclasses.dataclass(kw_only=True)
class LarryConditionalEncoderBlockModuleConfig(LLMBlockModuleConfig):
    d_cond: int | None = None
    is_causal: bool = False


@dataclasses.dataclass(kw_only=True)
class FlowMatchingConfig(ModelConfig):
    sigma_min: float = 1e-4
    t_sampling: Literal["uniform", "logit_normal"] = "logit_normal"
    t_mean: float = 0.0
    t_std: float = 1.0


@dataclasses.dataclass(kw_only=True)
class LarryRecurrentTrunkConfig(LLMBlockModuleConfig):
    n_layers: int = 4

    thought_init_std: float = 0.02  # Huginn uses sqrt(2/5) ≈ 0.6325
    thought_state_injection_type: Literal["sum", "concat"] = "sum"
    thought_state_init: Literal["none", "normal", "embed", "like-init", "zero", "unit"] = "like-init"
    exit_criteria: Literal["latent-diff", "logit-kl", "none"] = "latent-diff"
    logits_kl_exit_criteria_threshold: float = 5e-4
    latent_diff_exit_criteria_threshold: float = 0.03
    mean_thinking_steps: int = 32
    backprop_depth: int = 8

    iteration_projection_norm_type: NormTypeEnum = "rms"
    iteration_projection_norm_placement: Literal["pre", "post"] | None = None

    lockstep_n: bool = True
    lockstep_k: bool = True

    kv_cache_strategy: Literal["huginn", "per_iteration", "shared"] = "huginn"
    kv_cache_budget: int = 16
    kv_cache_max_iterations: int = 64


@dataclasses.dataclass(kw_only=True)
class VocabularyConfig(Config):
    vocab_size: int
