import dataclasses
from typing import Literal, Self, ClassVar

from larry.common.utils.registrable import Registrable
from larry.common.utils.types import TypeRegistry


@dataclasses.dataclass(kw_only=True)
class ModelConfig(Registrable, root=True):
    REGISTRY: ClassVar[TypeRegistry[ModelConfig]]

    d_model: int = 16
    n_layers: int = 6

    max_position_embeddings: int = 1024


@dataclasses.dataclass(kw_only=True)
class AttentionModuleConfig(ModelConfig):
    d_queries: int = 8
    d_values: int = 8
    n_query_groups: int = 1
    n_heads: int = 2
    use_qkv_bias: bool = False

    positional_encoding_name: Literal["rotary"] = "rotary"

    use_split_rope: bool = False
    split_rope_scale: float = 7.5
    rotary_embedding_dim: int = 4

    attention_dropout_p: float = 0.1
    proj_dropout_p: float = 0.1

    is_causal: bool = True

    @classmethod
    def from_child(cls, config: AttentionModuleConfig) -> Self:
        config_dict = dataclasses.asdict(config)
        dummy = dataclasses.asdict(cls())
        return cls(**{
            k: config_dict.get(k, v)
            for k, v in dummy.items()
            if k in config_dict.keys()
        })


@dataclasses.dataclass(kw_only=True)
class LLMBlockModuleConfig(AttentionModuleConfig):
    mlp_dropout_p: float = 0.1
    d_inner: int = 64
