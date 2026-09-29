# the larry initiative
the larry initiative is a project focused on producing as ethical AI as possible while performing as close to SOTA as 
possible.

Data used, redistributed, and/or produced for the larry initiative follow the following guidelines:
- Open license (PD, CC0-1.0, CC-BY)
  - Not SA, NC, or ND
- Not explicitly consent-revoked for redistribution or training (implicitly covered by licensing, but explicit consent
revocation is respected)
- Not generated, filtered, or otherwise scored or processed using a pretrained model whose:
  - Weights and code are not open
  - Training data doesn't abide by the above criteria

As an example, the entirety of libriheavy (~50k hours) was aligned to human-generated transcripts using 
Zengwei/icefall-asr-librispeech-zipformer-2023-05-15, a model trained entirely on librispeech, which is clean data by 
the above criteria. As such, libriheavy is entirely included in the larry initiative unaltered and as "seed data", while
 Zengwei/icefall-asr-librispeech-zipformer-2023-05-15 is included as a "seed model", whose training data is librispeech.
 Any larry models trained on libriheavy inherit this provenance.

As another example, libritts is a mostly speaker-disjoint from libriheavy librivox derivative. However, it was aligned 
by a proprietary, unreleased aligner by a Google research team. Additionally, the clips included in libritts are only 
those where the aligned produced a 0-edit error transcription to the ground truth on. As such, neither the procured set 
of voice audio clips nor the aligned transcriptions paired with such clips are allowable under this criteria. libritts-r
 also falls under this, but doubly-so because the 24khz audio samples are upsampled using an unreleased speech upsampler
 model.

Web-crawled data, proprietary data, data retrieved from social media or forum posts, data from blogs or public writing 
sites, and data otherwise collected without the human-generator's knowledge is prohibited within this initiative.

The intent is to:
1. Produce data and classifier, alignment, metric/rating, and generative models entirely from data that 
abides by the above criteria.
2. Release this data redistributed under the larry initiative (where the licenses or terms of use allow). 
3. Release models trained on this data as well for people (researchers and companies alike) to use for free and free 
from guilt.
4. Produce/filter abiding data using these models trained only on known-good data with pure provenance and distribute.
5. Train on this greater breadth and diversity of data, including aligned transcripts, captioned/recaptioned images, and 
 synthetic text or conversations.
6. Analyze and publish results; publish further generations of larry models.

Rinse and repeat.

All code within this project is human-made. Commercial generative AI is only used for feedback and aiding development 
through means that don't involve producing code to be committed to this project's repository.

# Seed Models Overview

## Text
N/A
## Voice
### Alignment/Transcriptions
- facebook/wav2vec2-large-960h-lv60-self
  - trained entirely on librivox/gutenberg data (citation needed)
- Zengwei/icefall-asr-librispeech-zipformer-2023-05-15
  - trained entirely on librispeech (citation needed)
## Image
N/A

# Seed Data Overview

## Text
TBD
## Voice
All voice data as redistributed (again, where applicable) or trained on is filtered using standard quality and <20s clip
 criteria for TTS and/or ASR. 
- Libriheavy
  - ~37k hours for training
  - ~7 hours for val
- Common Voice 27.0
  - ~2k hours for training
  - ~3.1 hours for val
- scotus
  - ~1.1k hours for training
  - ~6 hours for val
  - manually built by taking scotus proceedings audio and official government transcripts (PD0) and force-aligning using
 the above seed model aligner.
- VCTK
  - 24.9 hours for training
  - 1.2 hours for val
## Image
All image data was filtered by per-image licenses where possible.
- Wikimedia Commons
- Open Images
- VizWiz
- COCO 2017
- Cleveland CC0 print images
- Localized Narratives
- iNaturalist
- PD12M (images only; descriptions were produced with a poisoned model)
- Megalith-10M (images only; descriptions were produced with a poisoned model)
- Art institute of Chicago dump
- TextCaps, Crossmodal-3600
- CommonCatalog
- Te Papa
- NASA Image Library
- Rijkmuseum
- Dollar Street (images only; no descriptions exist)
- VisualGenome
- Museums Victoria of Melbourne

# larry Models
## Foundational/metric/backbone encoder/decoder models:
- larry-clip
    - replacement CLIP model
- larry-image-vae
    - replacement SDXL VAE model
- larry-speech-tokenizer
    - replacement CosyVoice2 encoder/speech tokenizer; FSQ bottleneck (6561 codes, 25 Hz), trained on CTC criteria
- larry-speech-decoder
    - replacement CosyVoice2 decoder
- larry-speaker-encoder
    - replacement CAM++ encoder
- larry-cfm
    - Flow-matching chunk-aware, causal CFM, tokens->mel, conditioned on prompt mel + speaker embedding
- larry-hift
    - replacement HiFT (HiFiGAN + iSTFT + NSF) vocoder (mel->waveform) model
- larry-tokenizer
    - own BPE text tokenizer, trained on all Larry text data including captions and transcriptions
- ~~larry-aligner~~
    - CTC forced aligner, trained on the model-free seed (Common Voice, VCTK, LJSpeech) to align long-form LibriVox 
against Gutenberg text
    - not needed anymore due to the above mentioned seed models

## Pretrained, single domain models using the recurrent trunk:
- larry-asr
    - replacement ASR model
- larry-tts
    - replacement TTS model
- larry-vits
    - replacement ViTS model
- larry-dit
    - replacement image generation model
    - will NOT be released due to abuse concerns (if it is even possible with the data I can procure)
- larry-base
  - text-only pretrained LLM
- larry-multimodal
  - text-only finetuned LLM

## The magnum opus:
- larry-multimodal-base
    - foundational world model combining larry-asr, larry-tts, larry-vits, and larry-dit into a single model with 
a shared recurrent trunk and per-modal encoders and decoders
    - pretrained on a superset of labeled voice, image, and raw text datasets, including instruct datasets in 
finetuning phase due to small size of model
- larry-multimodal-instruct
    - finetuned on more instruct datasets specifically

## Stretch goals:
- larry-multimodal-extras
    - attempt to freeze modal heads, add additional modalities, and finetune the trunk and new encoders/decoders on new 
datasets while maintaining performance on the original datasets
    - keep existing pretrain/finetune data in training as auxiliary loss to retain performance, but keep it sporadic so 
 the focus is on the new modality
    - tests retrainability to adapt to new modalities
    - first candidate is audio (not voice)
    - even further stretch candidates are music as separate from regular audio, actions, 3d models, and video


The intent is to train the models in the "Pretrained" section above first and initialize larry-multimodal-*'s 
encoder/decoder heads with these pretrained models' and let a new recurrent trunk learn to unify the modalities. Then, 
finetune larry-multimodal-* on instruct datasets to create MegaLarry-Instruct as separate. This will take months to 
years without rented gpu nodes.

# Other
## Requesting Data Removal
If you can prove you are the individual who generated a record or multiple records of data, or if you are depicted in a 
record or multiple records of data, and you wish for your data to be excluded, please provide your details in an email
to the lead developer, Mekadrom/Zadarimm/AHiggsBroson at `zadarimm@gmail.com`. No information will be shared, and the 
records affected will be taken down to the best of my ability either through republishing affected datasets or 
publishing a blacklist alongside the dataset and requiring all users of that data to filter using the blacklist.

## Contribution
don't, please. this is my baby, larry. you wouldn't make a change request to someone else's baby, would you?
