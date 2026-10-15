import dataclasses
from typing import Literal, Self, ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class ModelConfig(Registrable, root=True):
    """Superclass for configs for ``LarryModel`` implementations.

    Includes configuration for a maximum sequence length (since all ``LarryModel`` implementations work on sequential,
    varying length data) and a base d_model.
    """
    REGISTRY: ClassVar[TypeRegistry[ModelConfig]]

    d_model: int = 16

    max_position_embeddings: int = 1024


@dataclasses.dataclass(kw_only=True)
class AttentionModuleConfig(ModelConfig):
    """Config class for all ``LarryModel`` implementations which require configuring ``LarryAttention`` submodules."""
    n_query_groups: int = 1
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
        config_dict = dataclasses.asdict(config)
        dummy = dataclasses.asdict(cls())
        return cls(**{
            k: config_dict.get(k, v)
            for k, v in dummy.items()
            if k in config_dict.keys()
        })


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
