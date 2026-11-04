import dataclasses
from typing import Any

import pytest
import torch

from larry.multimodal.config.model.model_configs import LarryMultimodalModelConfig, TextModuleConfig
from larry.multimodal.data.dataloading.dataloading import PretrainingMultimodalCollator
from larry.multimodal.model.multimodal_model import LarryMultimodalModel
from tests.unit.multimodal.test_multimodal_generation import _vocab
from tests.unit.multimodal.test_multimodal_model import BLOCK, D_MODEL, _config

E = 16


def _text(config: LarryMultimodalModelConfig, embedding_dim: int | None) -> dict[str, Any]:
    return {
        "text_prelude_config": dataclasses.replace(config.text_prelude_config, embedding_dim=embedding_dim),
        "text_coda_config": dataclasses.replace(config.text_coda_config, embedding_dim=embedding_dim),
    }


def _model(collator: PretrainingMultimodalCollator, embedding_dim: int | None = E) -> LarryMultimodalModel:
    torch.manual_seed(0)
    base = _config(_vocab(collator))
    return LarryMultimodalModel(dataclasses.replace(base, **_text(base, embedding_dim))).eval()


def test_the_default_factorizes_at_a_quarter_of_the_production_width() -> None:
    config = TextModuleConfig(vocab_size=100, d_model=1024)

    assert config.embedding_dim == 256 and config.factorized and config.table_dim == 256


def test_a_factorized_table_projects_up_and_the_head_projects_back_down(
        collator: PretrainingMultimodalCollator,
) -> None:
    model = _model(collator)
    vocab = _vocab(collator)

    assert model.text_prelude.wte.weight.shape == (vocab, E)
    assert model.text_prelude.embed_proj is not None and model.text_prelude.embed_proj.weight.shape == (D_MODEL, E)
    assert model.text_coda.head_proj is not None and model.text_coda.head_proj.weight.shape == (E, D_MODEL)
    assert model.text_coda.lm_head.weight is model.text_prelude.wte.weight
    ids = torch.tensor([[1, 2, 3]])
    assert model.text_prelude.embed(ids).shape == (1, 3, D_MODEL)
    assert model.text_coda.head(torch.randn(1, 3, D_MODEL)).shape == (1, 3, vocab)


def test_disabling_it_embeds_at_d_model_without_projections(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator, embedding_dim=None)

    assert model.text_prelude.wte.weight.shape == (_vocab(collator), D_MODEL)
    assert model.text_prelude.embed_proj is None and model.text_coda.head_proj is None
    assert model.text_coda.lm_head.weight is model.text_prelude.wte.weight


def test_factorized_embeddings_start_at_the_unfactorized_scale(collator: PretrainingMultimodalCollator) -> None:
    factorized = _model(collator).text_prelude.embed(torch.arange(_vocab(collator)))
    plain = _model(collator, embedding_dim=None).text_prelude.embed(torch.arange(_vocab(collator)))

    assert factorized.std().item() == pytest.approx(plain.std().item(), rel=0.25)


def test_a_training_step_reaches_both_projections_and_the_tied_table(collator: PretrainingMultimodalCollator) -> None:
    model = _model(collator).train()
    ids = torch.tensor([[1, 2, 3, 4]])

    logits = model.text_coda.head(model.text_prelude.embed(ids))
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([2, 3, 4, 5])).backward()

    assert model.text_prelude.wte.weight.grad is not None
    assert model.text_prelude.embed_proj.weight.grad is not None  # type: ignore[union-attr]
    assert model.text_coda.head_proj.weight.grad is not None  # type: ignore[union-attr]


@pytest.mark.parametrize("embedding_dim", [None, D_MODEL, D_MODEL + 1])
def test_embedding_dims_that_would_not_shrink_the_table_embed_at_d_model(embedding_dim: int | None) -> None:
    config = TextModuleConfig(vocab_size=100, embedding_dim=embedding_dim, **BLOCK)

    assert not config.factorized and config.table_dim == D_MODEL


def test_a_non_positive_embedding_dim_is_refused() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        TextModuleConfig(vocab_size=100, embedding_dim=0, **BLOCK)


def test_the_prelude_and_coda_must_agree_on_the_embedding_dim(collator: PretrainingMultimodalCollator) -> None:
    base = _config(_vocab(collator))

    with pytest.raises(ValueError, match="embedding_dim must match"):
        dataclasses.replace(base, text_coda_config=dataclasses.replace(base.text_coda_config, embedding_dim=8))
