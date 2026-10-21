import dataclasses
from typing import Literal, Self

from larry.common.config.config import TaggedConfig


@dataclasses.dataclass(kw_only=True)
class ModelConfig(TaggedConfig):
    """Superclass for configs for ``LarryModel`` implementations.

    Includes configuration for a maximum sequence length (since all ``LarryModel`` implementations work on sequential,
    varying length data) and a base d_model.
    """

    d_model: int = 16

    max_position_embeddings: int = 1024


@dataclasses.dataclass(kw_only=True)
class AttentionModuleConfig(ModelConfig):
    """Config class for all ``LarryModel`` implementations which require configuring ``LarryAttention`` submodules."""

    n_query_groups: int = 16
    n_heads: int = 16
    use_qkv_bias: bool = False

    positional_encoding_name: Literal["rotary"] = "rotary"

    use_split_rope: bool = False
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

    mlp_dropout_p: float = 0.1


@dataclasses.dataclass(kw_only=True)
class FSMNAttentionConfig(AttentionModuleConfig):
    """Subclass implementation of ``AttentionModuleConfig``.

    For use in configuring the specialized implementation of ``LarryAttention``, ``FSMNAttention`` which incorporates a
    CNN FSMN memory branch, and overrides the default value for ``is_causal`` to False for bidirectional attention.
    """

    fsmn_kernel_size: int = 31
    is_causal: bool = False
