# the larry initiative
the larry initiative is a project focused on producing as ethical AI as possible while performing as close to SOTA as 
possible.

Data used, redistributed, and/or produced for the larry initiative follow the following guidelines:
- Open license (PD, CC0-1.0, occasionally CC-BY)
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
those where the aligner produced a 0-edit error transcription to the ground truth on. As such, neither the procured set 
of voice audio clips nor the aligned transcriptions paired with such clips are allowable under this criteria. libritts-r
 also falls under this, but doubly-so because the 24khz audio samples are upsampled using an unreleased speech upsampler
 model.

Web-crawled data, proprietary data, data retrieved from social media or forum posts, data from blogs or public writing 
sites, and data otherwise collected without the human-generator's knowledge is prohibited within this initiative.

"Scraping" in the form of treating web pages as a stand-in for a non-existent API is acceptable where that website and 
its data provably falls under the above criteria. Typically, this is only done for government data, as that is always 
Public Domain.

Basically, if something is labeled as Public Domain by either aging into it (enough time having passed since its 
creation) or by the creator intentionally submitting it to the Public Domain (CC0 license), then this project treats 
that data as belonging to humanity. As such, the generative and descriptive models produced by this project are for 
humanity, with a request to attribute usages of those models to this project for awareness. CC-BY (attribution license) 
is allowed when the sources are easily attributable, such as entire datasets or models, when very few rows require 
attribution, or when very many rows are attributable to a single licensee. This project would not be feasible without 
large open source projects on top of Public Domain data that require attribution such as redistributions of librivox/
librilight, HiFi-TTS, VCTK, etc.

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
- LJSpeech
  - 23.7 hours for training
- HiFi-TTS v1
  - 289.8 hours for training
  - 0.6 hours for val
## Image
All image data will be filtered by per-image licenses where necessary.

Pretrain:
- COCO 2017
- Cleveland CC0 print images
- Smithsonian Open Access (CC0 subset)
- Art institute of Chicago dump
- Te Papa
- NASA Image Library
- Rijkmuseum
- Museums Victoria of Melbourne
- The Met Open Access
- Library of Congress "Free to Use"
- NOAA, USGS, etc. (government image data)
- National Gallery of Art Open Data
- NYPL Public Domain Collections
- Paris Musées (CC0 subset)
- Statens Museum for Kunst
- Nationalmuseum Sweden
- Finnish National Gallery (CC0 subset)
- Walters Art Museum
- Yale Center for British Art / Yale University Art Gallery
- Princeton University Art Museum
- Getty Museum Open Content
- Nationaal Archief (CC0 subset)
- Europeana (PD/CC0 rights subset)
- US National Archives (NARA)
- DVIDS (public domain subset)
- NPS NPGallery, USFWS National Digital Library, USDA
- Landsat (USGS EarthExplorer)
- CDC PHIL (public domain subset)
- Custom Flickr crawl for PD/CC0 images, unfiltered
- ~~Open Images~~ - rows filtered by proprietary image models, which is excluded for now
- ~~Localized Narratives~~ - flickr crawl, individual images can be filtered
- ~~iNaturalist~~ - while the images are CC0 or CC-BY, the distribution prohibits commercial and non-open source 
derivatives
- ~~PD12M (images only; descriptions were produced with a poisoned model)~~ - rows filtered by clip, which is excluded 
for now
- ~~Megalith-10M (images only; descriptions were produced with a poisoned model)~~ - rows filtered by clip, which is 
excluded for now
- ~~TextCaps~~ - includes at least iNaturalist, excluded on that basis alone
- ~~CommonCatalog~~ - CC-BY on row-level; difficult to attribute
- ~~Dollar Street (images only; no descriptions exist)~~ - CC-BY-SA
- ~~VisualGenome~~ - YFCC100M derivative
- ~~Wikimedia Commons, filtered to PD/CC0~~ - got rate limited, pending communications
- ~~Bundesarchiv~~ - CC-BY-SA
- ~~ESA~~ - CC-BY-SA 3.0 IGO
- ~~Science Museum Group~~ - images are CC-BY-NC-SA
- ~~Harvard Art Museums~~ - images are not openly licensed
- ~~Brooklyn Museum~~ - mostly not CC0

Validation:
- Crossmodal-3600

Instruct:
- VizWiz

# larry Models
## Foundational/metric/backbone encoder/decoder models:
- larry-clip
    - replacement CLIP model
- larry-image-vae
    - replacement SDXL VAE model
- larry-speech-encoder
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
