import dataclasses
import difflib
import logging
from collections.abc import Callable

import numpy as np
import torch
import torchaudio.functional as AF
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from larry.voice.config.preprocessing.voice_mapper_configs import AlignerConfig, SpeechSegmentationAligningMapperConfig


@dataclasses.dataclass
class SpanScore:
    start: float
    end: float
    score: float
    count: int


class Aligner:
    def __init__(self, config: AlignerConfig, device="cuda", dtype=torch.float16):
        self.config = config

        self.frame_seconds = self.config.stride / self.config.sample_rate

        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_id)
        model = Wav2Vec2ForCTC.from_pretrained(
            config.model_id, dtype=self.dtype, mask_time_prob=0.0, mask_feature_prob=0.0
        )
        self.model = model.to(device=self.device).eval()  # pyright: ignore[reportCallIssue]

        vocab = self.processor.tokenizer.get_vocab()
        self.vocab = {k: v for k, v in vocab.items()}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # defaults are for facebook/wav2vec2-large-960h-lv60-self
        self.blank = self.vocab.get("<pad>", 0)
        self.delim = self.vocab.get("|", 4)

        self.log = logging.getLogger(__name__)

    def token_ids(self, word: str) -> list[int] | None:
        """Character ids for one normalized word; None if any character is out of vocab."""
        ids = []
        for c in word:
            i = self.vocab.get(c)
            if i is None:
                return None
            ids.append(i)
        return ids or None

    @torch.inference_mode()
    def ctc_emissions(
            self,
            waveform: torch.Tensor,
            progress: Callable[[int, int], None] | None = None
    ) -> torch.Tensor:
        """Stitched log-probs, [T, C] float32, one row per 20 ms frame. Processes only single examples, not batches."""
        win = int(self.config.chunk_seconds * self.config.sample_rate)
        trim = int(self.config.overlap_seconds * self.config.sample_rate)

        hop = win - 2 * trim
        if hop <= 0:
            raise ValueError("chunk_seconds must exceed 2 * overlap_seconds")
        if not self.model.config.vocab_size:
            raise ValueError("please load a model that configures a vocab_size on its config")

        trim_frames = trim // self.config.stride

        n_samples = len(waveform)
        total_frames = max(1, (n_samples - self.config.telomere_size) // self.config.stride + 1)

        # pre-init output tensor for vocab probs per frame for CTC
        out = torch.zeros((total_frames, self.model.config.vocab_size), dtype=torch.float32, device=self.device)
        filled = torch.zeros(total_frames, dtype=torch.bool, device=self.device)

        # collect a list of start indices by hopping
        starts = list(range(0, max(1, n_samples - 2 * trim), hop))
        for i, start in enumerate(starts):
            end = min(n_samples, start + win)
            seg = waveform[start:end]

            if len(seg) < 2 * self.config.telomere_size:
                continue

            # zero-mean unit-var norm, expected by at least facebook/wav2vec2-large-960h-lv60-self
            # do this in float32 before any downcasting
            x = seg.to(self.device).float()
            x = (x - x.mean()) / (x.std() + 1e-7)
            x = x.to(self.device, self.dtype)

            logits = self.model(x.unsqueeze(0)).logits.float()[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            frame_start = start // self.config.stride

            if i == 0:
                low = 0
            else:
                low = trim_frames

            high = log_probs.shape[0]

            align_low = frame_start + low
            align_high = min(total_frames, frame_start + high)
            align_dist = align_high - align_low

            if align_high > align_low:
                out[align_low:align_high] = log_probs[low:low + align_dist]
                filled[align_low:align_high] = True

            if progress is not None:
                progress(i + 1, len(starts))

        if not filled.all():
            # last partial frames can be unfilled
            # make them confident blanks so the trellis never has to path through uninitialized zeros.
            out[~filled] = -30.0  # set log prob of every position to large negative for softmax or greedy
            out[~filled, self.blank] = 0.0  # set log prob of just the blank vocab id to 0; effectively large confidence

        return out

    def greedy_words(self, emissions: torch.Tensor) -> list[tuple[str, int, int]]:
        """[(word, start_frame, end_frame)] from argmax - the anchor hypothesis."""
        best = emissions.argmax(dim=-1).tolist()

        words = []
        chars = []
        start = None
        prev = -1

        for i, token_id in enumerate(best):
            if token_id != prev:
                symbol = self.inv_vocab.get(token_id, "")
                if token_id != self.blank and symbol not in ("<s>", "</s>", "<unk>"):
                    if symbol == "|":
                        # emit word
                        if chars:
                            words.append(("".join(chars), start, i))
                            chars = []
                            start = None
                    else:
                        if start is None:
                            start = i
                        chars.append(symbol)
                prev = token_id
        if chars:
            words.append(("".join(chars), start, len(best)))
        return words

    def align_block(self, emissions: torch.Tensor, words: list[str]) -> list[SpanScore | None]:
        """Exact forced alignment of `words` inside one emission slice.

        Returns [(start_frame, end_frame, score)] per word, frames relative to the slice.
        Raises RuntimeError if the block has too few frames for the target.
        """
        targets = []
        owner = []
        for i, word in enumerate(words):
            ids = self.token_ids(word)
            if ids is None:
                continue
            if targets:
                targets.append(self.delim)
                owner.append(-1)
            targets.extend(ids)
            # track which word each ctc alignment frame is "owned" by
            owner.extend([i] * len(ids))

        if not targets:
            return [None] * len(words)

        log_probs = emissions.unsqueeze(0)
        ctc_targets = torch.tensor([targets], dtype=torch.int32, device=emissions.device)

        labels, scores = AF.forced_align(log_probs, ctc_targets, blank=self.blank)
        spans = AF.merge_tokens(labels[0], scores[0], blank=self.blank)

        if len(spans) != len(targets):
            raise ValueError(f"forced_align returned {len(spans)} spans for {len(targets)} targets")

        # use list instead of tuple for mutability
        out: list[SpanScore | None] = [None] * len(words)
        for span, word_idx in zip(spans, owner):
            if word_idx < 0:
                continue

            cur = out[word_idx]
            start = span.start
            end = span.end
            score = float(span.score)

            if cur is None:
                out[word_idx] = SpanScore(start, end, score, 1)
            else:
                cur.start = min(cur.start, start)
                cur.end = max(cur.end, end)
                cur.score += score
                cur.count += 1

        return out


@dataclasses.dataclass
class LongAlignmentResult:
    spans: list[SpanScore | None]
    n_failed_words: int
    greedy_decode_score: float


class LongAlignment:
    def __init__(
            self,
            config: SpeechSegmentationAligningMapperConfig,
            aligner: Aligner,
            emissions: torch.Tensor,
            ref_words: list[str],
    ) -> None:
        self.config = config
        self.aligner = aligner
        self.emissions = emissions
        self.ref_words = ref_words

    def run(self) -> LongAlignmentResult:
        block_seconds = self.config.block_seconds
        max_block_seconds = self.config.max_block_seconds
        slack_seconds = self.config.slack_seconds
        frame_seconds = self.aligner.config.stride / self.aligner.config.sample_rate

        n_frames = len(self.emissions)
        block_frames = int(block_seconds / frame_seconds)
        max_block_frames = int(max_block_seconds / frame_seconds)
        slack = int(slack_seconds / frame_seconds)

        hyp_words = self.aligner.greedy_words(self.emissions)
        mblocks = self._match_blocks(hyp_words)
        anchors = self._choose_anchors(
            hyp_words,
            n_frames,
            block_frames,
            blocks=mblocks
        )

        bounds = [(0, 0)] + anchors + [(len(self.ref_words), n_frames)]
        blocks = [(bounds[i][0], bounds[i + 1][0], bounds[i][1], bounds[i + 1][1])
                  for i in range(len(bounds) - 1)]
        blocks = [b for b in blocks if b[1] > b[0] and b[3] > b[2]]
        blocks = self._split_long(blocks, max_block_frames)

        spans: list[SpanScore | None] = [None] * len(self.ref_words)
        failed_words = 0
        for w0, w1, f0, f1 in blocks:
            lo = max(0, f0 - slack)
            hi = min(n_frames, f1 + slack)
            sub = self.emissions[lo:hi]
            try:
                got = self.aligner.align_block(sub, self.ref_words[w0:w1])
            except (RuntimeError, ValueError):
                self.aligner.log.exception(
                    f"len(ref_words)={len(self.ref_words)}, n_frames*0.02={n_frames * 0.02} "
                    f"block words {w0}-{w1} frames {lo}-{hi} failed:"
                )
                failed_words += w1 - w0
                continue
            for k, span_score in enumerate(got):
                if span_score is None:
                    failed_words += 1
                    continue
                spans[w0 + k] = SpanScore(
                    (lo + span_score.start) * frame_seconds,
                    (lo + span_score.end) * frame_seconds,
                    span_score.score,
                    span_score.count,
                )

        greedy_decode_score = float(1.0 - sum(b.size for b in mblocks) / max(1, len(self.ref_words)))

        if greedy_decode_score > 0.5:
            self.aligner.log.info(
                f"poor score ({greedy_decode_score}): hyp_words={hyp_words[:30]}, ref_words={self.ref_words[:30]}")

        return LongAlignmentResult(spans, int(failed_words), greedy_decode_score)

    def _choose_anchors(
            self,
            hyp_words: list[tuple[str, int, int]],
            n_frames: int,
            block_frames=3000,
            blocks=None
    ):
        max_chars_per_frame = self.config.max_chars_per_frame
        min_anchor_run = self.config.min_anchor_run

        if blocks is None:
            blocks = self._match_blocks(hyp_words)
        candidates = []
        for h0, r0, size in blocks:
            if size < min_anchor_run:
                continue
            # Skip the first and last of the run; anchor on each interior word.
            for k in range(1, size - 1):
                candidates.append((r0 + k, hyp_words[h0 + k][1]))
        if not candidates:
            return []

        # Prefix sum of characters so the feasibility test below is O(1) per candidate.
        cum = [0]
        for w in self.ref_words:
            cum.append(cum[-1] + len(w) + 1)

        anchors = []
        last_frame = 0
        last_ref = 0
        total = cum[-1]
        for ref_i, frame in candidates:
            if ref_i <= last_ref or frame <= last_frame:
                continue
            if frame - last_frame < block_frames and anchors:
                continue
            if cum[ref_i] - cum[last_ref] > (frame - last_frame) * max_chars_per_frame:
                continue
            # the text still to come must also fit in the frames still to come
            if total - cum[ref_i] > (n_frames - frame) * max_chars_per_frame:
                continue
            anchors.append((ref_i, frame))
            last_frame = frame
            last_ref = ref_i
        return anchors

    def _split_long(self, blocks: list[tuple[int, int, int, int]], max_block_frames) -> list[tuple[int, int, int, int]]:
        out = []
        for w0, w1, f0, f1 in blocks:
            n_frames = f1 - f0
            if n_frames <= max_block_frames or w1 - w0 < 2:
                out.append((w0, w1, f0, f1))
                continue
            parts = int(np.ceil(n_frames / max_block_frames))
            lens = np.array([len(self.ref_words[i]) + 1 for i in range(w0, w1)], dtype=np.float64)
            cum = np.concatenate([[0.0], np.cumsum(lens)])
            base_w, base_f = w0, f0
            for p in range(parts):
                last = p == parts - 1
                target = cum[-1] * (p + 1) / parts
                if last:
                    wi = w1
                    fi = f1
                else:
                    wi = min(max(base_w + int(np.searchsorted(cum, target)), w0 + 1), w1)
                    fi = base_f + int(round(n_frames * (p + 1) / parts))
                if wi > w0:
                    out.append((w0, wi, f0, fi))
                w0, f0 = wi, fi
                if w0 >= w1:
                    break
        return out

    def _match_blocks(self, hyp_words: list[tuple[str, int, int]]) -> list[difflib.Match]:
        hyp = [w for w, _, _ in hyp_words]
        return difflib.SequenceMatcher(None, hyp, self.ref_words, autojunk=False).get_matching_blocks()


class TokenTimeline:
    spans: list[SpanScore | None]
    ok_mask: list[bool]

    def __init__(self, times: list[SpanScore | None], groups: list[int]) -> None:
        self.times = times
        self.groups = groups

    def fill_gaps(self) -> None:
        n_spans = len(self.times)
        self.spans = [None] * n_spans
        self.ok_mask = [t is not None for t in self.times]

        i = 0
        while i < n_spans:
            j = i
            while j < n_spans and self.groups[j] == self.groups[i]:
                j += 1

            idx = [
                k
                for k in range(i, j)
                if self.times[k] is not None
            ]

            if idx:
                for k in range(i, j):
                    cur_time = self.times[k]
                    if cur_time is not None:
                        self.spans[k] = cur_time
                        continue

                    prev = max((p for p in idx if p < k), default=None)
                    nxt = min((p for p in idx if p > k), default=None)

                    start = None
                    end = None
                    if prev is None:
                        if nxt is not None:
                            span = self.times[nxt]
                            if span:
                                start = span.start
                                end = span.start
                    elif nxt is None:
                        if prev is not None:
                            span = self.times[prev]
                            if span:
                                start = span.end
                                end = span.end
                    else:
                        a = None
                        p = self.times[prev]
                        if p is not None:
                            a = p.end

                        b = None
                        n = self.times[nxt]
                        if n is not None:
                            b = n.start

                        if a is not None and b is not None:
                            frac = (k - prev) / (nxt - prev)
                            start = a + (b - a) * frac
                            end = a + (b - a) * min(1.0, frac + 1.0 / (nxt - prev))

                    if start is not None and end is not None:
                        self.spans[k] = SpanScore(start, max(start, end), 0.0, 1)
                    else:
                        raise ValueError(f"could not determine start and end for filled gap")
            i = j

        # repair
        last = -1e9
        for k in range(n_spans):
            span = self.spans[k]
            if span is None:
                continue
            span.start = max(span.start, last)
            span.end = max(span.end, span.start)
            last = span.end
