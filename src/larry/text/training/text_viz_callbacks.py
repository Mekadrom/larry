from abc import ABC

from larry.common.training import trainers
from larry.text.training import text_trainers


class TextVizCallback(trainers.VizCallback, ABC):
    ...


class SmokeTestVizCallback(TextVizCallback):
    prompts: list[str] = [
        "To be or not to be, ",
        "Doth mother know you weareth her "
    ]

    def viz(self, trainer: trainers.TrainerBase, step: int) -> None:
        if not isinstance(trainer, text_trainers.SmokeTestLLMTrainer):
            raise ValueError("SmokeTestVizCallback needs a SmokeTestLLMTrainer")

        model = trainer.accelerator.unwrap_model(trainer.model)
        model.eval()

        if trainer.tokenizer is None:
            self.log.error("Tokenizer was not loaded.")
            return

        inputs = trainer.tokenizer(
            self.prompts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False
        )

        output = model.generate(
            inputs["input_ids"].to(model.device),
            attention_mask=inputs["attention_mask"].to(model.device),
            num_samples=2,
            max_new_tokens=100,
            eos_token_id=trainer.tokenizer.eos_token_id,
            pad_token_id=trainer.tokenizer.pad_token_id,
            temperature=1.0
        )
        full_text = trainer.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        for i, text in enumerate(full_text):
            trainer.metrics.add_text(f"eval/text_continuation/example_{i}", text, step)
        model.train()
