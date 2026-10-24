import dataclasses
from typing import Literal

import torch
from torch import nn

from larry.common.model.attention import LarryAttentionOutput
from larry.common.model.kv_cache import LarryMultimodalModelKVCache
from larry.common.model.recurrent import LarryRecurrentTrunk
from larry.common.training.training_model import LarryModel
from larry.image.model.image_modules import Patchify, Unpatchify
from larry.multimodal.config.model.model_configs import LarryMultimodalModelConfig, TextModuleConfig, ImageModuleConfig, \
    VoiceModuleConfig
from larry.multimodal.processor.processor import ImagePlaceholder, VoicePlaceholder
from larry.voice.model.speech_decoder import LarrySimpleEncoderBlock


@dataclasses.dataclass(kw_only=True)
class LarryMultimodalModelOutput(LarryAttentionOutput[LarryMultimodalModelKVCache]):
    attention_mask: torch.Tensor
    image_embeddings: torch.Tensor | None = None
    voice_embeddings: torch.Tensor | None = None


class TextPrelude(nn.Module):
    def __init__(self, config: TextModuleConfig) -> None:
        super().__init__()

        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        for block in self.blocks:
            x = block(hidden_states=x)
        return x


class ImagePrelude(nn.Module):
    def __init__(self, config: ImageModuleConfig) -> None:
        super().__init__()

        self.conv_frontend = nn.Sequential(
            nn.Conv2d(3, config.d_model, kernel_size=3, stride=1, padding=1)
        )
        self.patchify = Patchify(config.patch_size)
        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_frontend(x)  # (B, d_model, H, W)
        x = self.patchify(x).transpose(-2, -1)  # (B, H//patch_size*W//patch_size, d_model*patch_size*patch_size)
        for block in self.blocks:
            x = block(hidden_states=x)
        return x


class VoicePrelude(nn.Module):
    def __init__(self, config: VoiceModuleConfig) -> None:
        super().__init__()

        self.ute = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(config.n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ute(x)
        for block in self.blocks:
            x = block(hidden_states=x)
        return x


class TextCoda(nn.Module):
    def __init__(self, config: TextModuleConfig) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(config.n_layers)
        ])
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(hidden_states=x)
        x = self.lm_head(x)
        return x


class ImageCoda(nn.Module):
    def __init__(self, config: ImageModuleConfig) -> None:
        super().__init__()

        self.unpatchify = Unpatchify(config.patch_size)
        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(config.n_layers)
        ])
        self.conv_refine = nn.Sequential(
            nn.Conv2d(3, config.d_model, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(hidden_states=x)
        x = self.unpatchify(x.transpose(-2, -1))  # (B, H//patch_size*W//patch_size, d_model*patch_size*patch_size)
        x = self.conv_refine(x)  # (B, 3, H, W)
        return x


class VoiceCoda(nn.Module):
    def __init__(self, config: VoiceModuleConfig) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            LarrySimpleEncoderBlock(config)
            for _ in range(config.n_layers)
        ])
        self.vm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(hidden_states=x)
        x = self.vm_head(x)
        return x


class LarryMultimodalModel(LarryModel[LarryMultimodalModelConfig]):
    def __init__(self, config: LarryMultimodalModelConfig) -> None:
        super().__init__(config)

        self._register_preludes()
        self.recurrent_block = LarryRecurrentTrunk(config.recurrent_trunk_config)
        self._register_codas()

    def _register_preludes(self) -> None:
        self.text_prelude = TextPrelude(self.config.text_prelude_config)

        self.image_prelude = None
        if self.config.image_prelude_config is not None:
            self.image_prelude = ImagePrelude(self.config.image_prelude_config)

        self.voice_prelude = None
        if self.config.voice_prelude_config is not None:
            self.voice_prelude = VoicePrelude(self.config.voice_prelude_config)

    def _register_codas(self) -> None:
        self.text_coda = TextCoda(self.config.text_coda_config)

        self.image_coda = None
        if self.config.image_coda_config is not None:
            self.image_coda = ImageCoda(self.config.image_coda_config)

        self.voice_coda = None
        if self.config.voice_coda_config is not None:
            self.voice_coda = VoiceCoda(self.config.voice_coda_config)

    def forward(
            self,
            modality_ids: torch.Tensor,
            input_ids: torch.Tensor,
            images: torch.Tensor | None,
            voice_unit_ids: torch.Tensor | None,
            attention_mask: torch.Tensor | None,
            kv_cache: LarryMultimodalModelKVCache | None,
            recurrent_kv_cache_strategy: Literal["huginn", "per_iteration", "shared"] = "huginn",
            use_cache: bool = False,
    ) -> LarryMultimodalModelOutput:
        input_embeddings = self.text_prelude(input_ids)

        images_mask = modality_ids == ImagePlaceholder.modality_id
        voices_mask = modality_ids == VoicePlaceholder.modality_id

        grid_mask = torch.ones(self.config.max_seq_len, self.config.max_seq_len)
        if attention_mask is None:
            attention_mask = torch.tril(grid_mask)
            attention_mask[images_mask] = 1

        if use_cache and kv_cache is None:
            kv_cache = LarryMultimodalModelKVCache(recurrent_kv_cache_strategy=recurrent_kv_cache_strategy)

        m_rope_dim_mask = torch.tensor(grid_mask)
        m_rope_dim_mask[voices_mask] = 2
        m_rope_dim_mask[images_mask] = 3

        image_embeddings = None
        if images is not None and self.image_prelude is not None:
            image_prelude_kv_cache = kv_cache.image_prelude_cache if kv_cache is not None else None
            image_embeddings = self.image_prelude(images, kv_cache=image_prelude_kv_cache, use_cache=use_cache)
            input_embeddings[images_mask] = image_embeddings

        voice_embeddings = None
        if voice_unit_ids is not None and self.voice_prelude is not None:
            voice_prelude_kv_cache = kv_cache.voice_prelude_cache if kv_cache is not None else None
            voice_embeddings = self.voice_prelude(voice_unit_ids, kv_cache=voice_prelude_kv_cache, use_cache=use_cache)
            input_embeddings[voices_mask] = voice_embeddings

        final_thought_state = self.recurrent_block(input_embeddings, attention_mask=attention_mask)

        text_logits = self.text_coda(final_thought_state[~(images_mask | voices_mask)])

        if images is not None and self.image_coda is not None:
            image_coda_kv_cache = kv_cache.image_coda_cache if kv_cache is not None else None
            image_embeddings = final_thought_state[images_mask]
            image_embeddings = self.image_coda(image_embeddings, kv_cache=image_coda_kv_cache, use_cache=use_cache)

        if voice_unit_ids is not None and self.voice_coda is not None:
            voice_coda_kv_cache = kv_cache.voice_coda_cache if kv_cache is not None else None
            voice_embeddings = final_thought_state[voices_mask]
            voice_embeddings = self.voice_coda(voice_embeddings, kv_cache=voice_coda_kv_cache, use_cache=use_cache)

        return LarryMultimodalModelOutput(
            logits=text_logits,
            attention_mask=attention_mask,
            image_embeddings=image_embeddings,
            voice_embeddings=voice_embeddings,
            kv_cache=kv_cache,
        )
