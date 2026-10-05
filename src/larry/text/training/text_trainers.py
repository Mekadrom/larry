import functools
import math
from abc import ABC
from typing import Mapping, Any

import numpy as np
import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from larry.common.config.criteria.criterion_configs import TeacherForcedVocabularyCrossEntropyCriterionConfig
from larry.common.criteria.criterion import Criterion
from larry.common.criteria.teacher_forced_vocabulary_ce_criterion import TeacherForcedVocabularyCrossEntropyCriterion
from larry.common.metrics.metrics import Metrics, AggregateMetricEntry, MetricDict
from larry.common.training.trainers import LarryTrainer, VizCallback, StepOutput
from larry.text.config.model.text_model_configs import SmokeTestLLMModelConfig
from larry.text.config.training.text_trainer_configs import TextTrainerConfig, SmokeTestLLMTrainerConfig
from larry.text.model.text_models import SmokeTestLLM, TextModel
from larry.text.training.text_viz_callbacks import SmokeTestVizCallback


class TextTrainer[M: TextModel, C: TextTrainerConfig = TextTrainerConfig](LarryTrainer[M, C], ABC):
    @torch.no_grad()
    def evaluate(self) -> torch.Tensor:
        loss = super().evaluate()
        self.metrics.add_scalar("eval/ppl", math.exp(loss.item()), self.train_state.global_step)
        self.metrics.add_scalar("eval/bpb", loss.item() / math.log(2), self.train_state.global_step)
        return loss

    def train_batch_size(self, batch: Mapping[str, Any]) -> int:
        return self._batch_size(batch)

    def eval_batch_size(self, batch: Mapping[str, Any]) -> int:
        return self._batch_size(batch)

    @staticmethod
    def _batch_size(batch: Mapping[str, Any]) -> int:
        return batch["input_ids"].shape[0]


class BlockDataset(Dataset):
    # noinspection PyMissingConstructor
    def __init__(self, tokens: np.ndarray, block_size: int) -> None:
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        start = i * self.block_size
        chunk = self.tokens[start:start + self.block_size + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class SmokeTestLLMTrainer(TextTrainer[SmokeTestLLM, SmokeTestLLMTrainerConfig]):
    def __init__(
            self,
            model: SmokeTestLLM,
            config: SmokeTestLLMTrainerConfig,
            metrics: Metrics,
            resume_dir: str | None = None
    ) -> None:
        super().__init__(model, config, metrics, resume_dir=resume_dir)
        self.tokens_seen = AggregateMetricEntry(tag="train/total_tokens", average_reduce="none")

    @staticmethod
    def build_token_stream(rows: list[list[int]], eos_id: int) -> np.ndarray:
        return np.concatenate([np.asarray(r + [eos_id], dtype=np.int32) for r in rows])

    @functools.cached_property
    def tokenizer(self) -> PreTrainedTokenizerBase | None:
        if isinstance(self.model_config, SmokeTestLLMModelConfig):
            return AutoTokenizer.from_pretrained(self.model_config.tokenizer_name)
        return None

    def train_forward_pass(self, batch: Mapping[str, Any]) -> StepOutput:
        if self.tokenizer is not None:
            self.tokens_seen.add_((batch["input_ids"] != self.tokenizer.pad_token_id).sum().item())
        return self._forward_pass(batch)

    def on_step_end(self, train_loss: torch.Tensor, step: int) -> MetricDict:
        if step % self.config.logging_steps == 0:
            self.metrics.add_scalar("train/loss_by_tokens", train_loss.item(), int(self.tokens_seen.value))
        return self._callback_func(lambda c: c.on_step_end, train_loss, step)

    def make_dataset_dict(self) -> dict[str, Dataset]:
        if self.tokenizer is None:
            raise ValueError(f"Please define a tokenizer for {type(self).__name__}")

        eos_id = self.tokenizer.eos_token_id
        if eos_id is None:
            eos_id = self.tokenizer.sep_token_id
        if not isinstance(eos_id, int):
            raise ValueError(f"expected an int EOS/SEP token id, got {eos_id!r}")

        dataset_dict = super().make_dataset_dict()

        train_rows: list[list[int]] = []
        for row in dataset_dict["train"]:
            if isinstance(row, Mapping):
                train_rows.append(row["input_ids"])

        val_rows: list[list[int]] = []
        for row in dataset_dict["validation"]:
            if isinstance(row, Mapping):
                val_rows.append(row["input_ids"])

        block_size = self.model_config.max_position_embeddings
        train_tokens = self.build_token_stream(train_rows, eos_id=eos_id)
        val_tokens = self.build_token_stream(val_rows, eos_id=eos_id)
        return {"train": BlockDataset(train_tokens, block_size), "validation": BlockDataset(val_tokens, block_size)}

    def make_viz_callback(self, viz_steps: int) -> VizCallback:
        return SmokeTestVizCallback(viz_steps)

    def make_criterion(self) -> Criterion:
        return TeacherForcedVocabularyCrossEntropyCriterion(TeacherForcedVocabularyCrossEntropyCriterionConfig())
