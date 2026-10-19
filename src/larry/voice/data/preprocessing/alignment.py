import bisect
import dataclasses
import difflib
import itertools
import logging
from collections.abc import Collection

import numpy as np
import torch
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from larry.voice.config.preprocessing.voice_configs import AlignerConfig
from larry.voice.config.preprocessing.voice_mapper_configs import SpeechSegmentationAligningMapperConfig


@dataclasses.dataclass
class SpanScore:
    """Helper dataclass for storing all information required to slice a span of audio."""

    start: float
    end: float
    score: float
    count: int


class Aligner:
    """Set of helper/utility functions wrapping a pretrained alignment/CTC model for aligning text and speech."""

    def __init__(
            self,
            config: AlignerConfig,
            device: str | int | torch.device = "cuda",
            dtype: torch.dtype = torch.float16
    ) -> None:
        """Loads the alignment model and processor."""
        self.log = logging.getLogger(__name__)
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
    def ctc_emissions(self, waveform: torch.Tensor) -> torch.Tensor:
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
        # inside ctc_emissions, replacing the per-chunk loop
        full = [
            s
            for s in starts
            if s + win <= n_samples
        ]
        tail = [
            s
            for s in starts
            if s + win > n_samples and n_samples - s >= 2 * self.config.telomere_size
        ]

        batch_size = self.config.emission_batch_size
        for b in range(0, len(full), batch_size):
            group = full[b:b + batch_size]
            chunks = [
                waveform[s:s + win]
                for s in group
            ]

            x = torch.stack(chunks).to(self.device).float()
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-7)

            logits = self.model(x.to(self.dtype)).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            for s, chunk_log_probs in zip(group, log_probs, strict=True):
                self._place(out, filled, s, chunk_log_probs, trim_frames, total_frames)

        for s in tail:
            x = waveform[s:n_samples].to(self.device).float()
            x = (x - x.mean()) / (x.std() + 1e-7)

            logits = self.model(x.to(self.dtype).unsqueeze(0)).logits.float()[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            self._place(out, filled, s, log_probs, trim_frames, total_frames)

        if not filled.all():
            # last partial frames can be unfilled
            # make them confident blanks so the trellis never has to path through uninitialized zeros.
            out[~filled] = -30.0  # set log prob of every position to large negative for softmax or greedy
            out[~filled, self.blank] = 0.0  # set log prob of just the blank vocab id to 0; effectively large confidence

        return out

    def _place(
            self,
            out: torch.Tensor,
            filled: torch.Tensor,
            start: int,
            log_probs: torch.Tensor,
            trim_frames: int,
            total_frames: int,
    ) -> None:
        """Helper for offsetting an audio span's bounds. Mutates the ``out`` and ``filled`` parameters."""
        low = 0 if start == 0 else trim_frames

        frame_start = start // self.config.stride
        high = min(total_frames, frame_start + log_probs.shape[0])
        if high <= frame_start + low:
            return

        count = high - frame_start - low
        out[frame_start + low:high] = log_probs[low:low + count]
        filled[frame_start + low:high] = True

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

    def align_block(
            self,
            emissions: torch.Tensor,
            words: list[str],
            wildcards: Collection[int] = (),
    ) -> list[SpanScore | None]:
        """Exact forced alignment of `words` inside one emission slice.

        Returns [(start_frame, end_frame, score)] per word, frames relative to the slice.
        Raises RuntimeError if the block has too few frames for the target.
        """
        if wildcards:
            try:
                return self._forced_spans(emissions, words, frozenset(wildcards))
            except (RuntimeError, ValueError):
                self.log.debug("block too short for its wildcards; aligning without them")
        return self._forced_spans(emissions, words, frozenset())

    def wildcard_emissions(self, emissions: torch.Tensor) -> torch.Tensor:
        return emissions.max(dim=-1).values - self.config.wildcard_penalty

    def _forced_spans(
            self,
            emissions: torch.Tensor,
            words: list[str],
            wildcards: frozenset[int],
    ) -> list[SpanScore | None]:
        wildcard = emissions.shape[1] if wildcards else None
        targets, owners = self.get_targets_owners(words, wildcards, wildcard)

        if not any(owner >= 0 for owner in owners):
            return [None] * len(words)

        if wildcard is not None:
            emissions = torch.cat([emissions, self.wildcard_emissions(emissions).unsqueeze(-1)], dim=-1)

        log_probs = emissions.unsqueeze(0)
        ctc_targets = torch.tensor([targets], dtype=torch.int32, device=emissions.device)

        labels, scores = AF.forced_align(log_probs, ctc_targets, blank=self.blank)
        spans = AF.merge_tokens(labels[0], scores[0], blank=self.blank)

        if len(spans) != len(targets):
            raise ValueError(f"forced_align returned {len(spans)} spans for {len(targets)} targets")

        # use list instead of tuple for mutability
        out: list[SpanScore | None] = [None] * len(words)
        for span, word_idx in zip(spans, owners, strict=True):
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

    def get_targets_owners(
            self,
            words: list[str],
            wildcards: Collection[int] = (),
            wildcard: int | None = None,
    ) -> tuple[list[int], list[int]]:
        """Calculates the owner word indices for each token (character) position index."""
        targets = []
        owners = []
        pending_wildcard = False
        for i, word in enumerate(words):
            pending_wildcard = pending_wildcard or (wildcard is not None and i in wildcards)
            ids = self.token_ids(word)
            if ids is None:
                continue

            if targets and targets[-1] != wildcard:
                targets.append(self.delim)
                owners.append(-1)

            if pending_wildcard:
                targets.append(wildcard)
                owners.append(-1)
                pending_wildcard = False

            targets.extend(ids)
            # track which word each ctc alignment frame is "owned" by
            owners.extend([i] * len(ids))

        if targets and wildcard is not None and (pending_wildcard or len(words) in wildcards):
            targets.extend([self.delim, wildcard])
            owners.extend([-1, -1])

        return targets, owners

    @torch.inference_mode()
    def clip_emissions(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        """Performs the forward pass through the alignment model to obtain CTC prediction logits."""
        normalized = []
        for w in waveforms:
            w = w.float()
            normalized.append((w - w.mean()) / (w.std() + 1e-7))

        sample_counts = [
            len(w)
            for w in waveforms
        ]
        lengths = torch.tensor(sample_counts)

        x = pad_sequence(normalized, batch_first=True).to(self.device, self.dtype)
        positions = torch.arange(x.shape[1])
        mask = (positions[None, :] < lengths[:, None]).long().to(self.device)

        logits = self.model(x, attention_mask=mask).logits.float()
        log_probs = torch.log_softmax(logits, dim=-1).cpu()
        # noinspection PyProtectedMember
        frame_counts = self.model._get_feat_extract_output_lengths(lengths)

        return [
            log_probs[i, :int(frame_counts[i])]
            for i in range(len(waveforms))
        ]

    def path_score(self, emissions: torch.Tensor, words: list[str]) -> float | None:
        """Log-prob of the best path forced through `words` over every frame of `emissions`."""
        targets, _ = self.get_targets_owners(words)
        if len(targets) == 0 or len(targets) > emissions.shape[0]:
            return None

        ctc_targets = torch.tensor([targets], dtype=torch.int32, device=emissions.device)
        try:
            _, scores = AF.forced_align(emissions.unsqueeze(0), ctc_targets, blank=self.blank)
        except RuntimeError:
            return None
        return float(scores[0].sum())

    def mismatch_score(self, emissions: torch.Tensor, forced: float) -> float:
        """A cost function for a given set of logits and the forced alignment target for a single position."""
        free = float(emissions.max(dim=-1).values.sum())
        return (free - forced) / emissions.shape[0]


@dataclasses.dataclass
class LongAlignmentResult:
    """Helper dataclass for the result of the alignment of an entire transcript to its entire audio."""

    spans: list[SpanScore | None]
    n_failed_words: int
    greedy_decode_score: float
    hyp_words: list[tuple[str, int, int]]
    unspoken: list[bool]


type AlignmentBlock = tuple[list[int], int, int]


class LongAlignment:
    """Helper wrapper around ``Aligner`` for aligning a single, long audio file with a long transcript.

    Produces ``SpanScore`` objects for the aligned parts of speech within the longer input.
    """

    def __init__(
            self,
            config: SpeechSegmentationAligningMapperConfig,
            aligner: Aligner,
            emissions: torch.Tensor,
            ref_words: list[str],
            turn_starts: Collection[int] = (),
    ) -> None:
        """Simple init."""
        self.config = config
        self.aligner = aligner
        self.emissions = emissions
        self.ref_words = ref_words
        self.turn_starts = frozenset(turn_starts)
        self.cum_chars = np.concatenate([[0], np.cumsum([len(w) + 1 for w in ref_words], dtype=np.int64)])

    def run(self, reuse_result: LongAlignmentResult | None = None) -> LongAlignmentResult:
        """Runs alignment on the entire speech segment in blocks, producing aligned spans using anchor alignments."""
        block_seconds = self.config.block_seconds
        max_block_seconds = self.config.max_block_seconds
        slack_seconds = self.config.slack_seconds
        frame_seconds = self.aligner.config.stride / self.aligner.config.sample_rate

        n_frames = len(self.emissions)
        block_frames = int(block_seconds / frame_seconds)
        max_block_frames = int(max_block_seconds / frame_seconds)
        slack = int(slack_seconds / frame_seconds)

        hyp_words = reuse_result.hyp_words if reuse_result else self.aligner.greedy_words(self.emissions)
        mblocks = self._match_blocks(hyp_words)
        anchors, unspoken = self._choose_anchors(
            hyp_words,
            n_frames,
            block_frames,
            blocks=mblocks
        )

        bounds = [(0, 0)] + anchors + [(len(self.ref_words), n_frames)]
        blocks: list[AlignmentBlock] = []
        for (w0, f0), (w1, f1) in itertools.pairwise(bounds):
            indices = [k for k in range(w0, w1) if k not in unspoken]
            if indices and f1 > f0:
                blocks.append((indices, f0, f1))

        blocks = self._split_long(blocks, max_block_frames)

        spans: list[SpanScore | None] = [None] * len(self.ref_words)
        failed_words = 0
        for indices, f0, f1 in blocks:
            lo = max(0, f0 - slack)
            hi = min(n_frames, f1 + slack)
            sub = self.emissions[lo:hi]
            try:
                got = self.aligner.align_block(sub, [self.ref_words[k] for k in indices], self._wildcards(indices))
            except (RuntimeError, ValueError):
                self.aligner.log.exception(
                    f"len(ref_words)={len(self.ref_words)}, n_frames*0.02={n_frames * 0.02} "
                    f"block words {indices[0]}-{indices[-1]} frames {lo}-{hi} failed:"
                )
                failed_words += len(indices)
                continue

            for k, span_score in zip(indices, got, strict=True):
                if span_score is None:
                    failed_words += 1
                    continue

                spans[k] = SpanScore(
                    (lo + span_score.start) * frame_seconds,
                    (lo + span_score.end) * frame_seconds,
                    span_score.score,
                    span_score.count,
                )

        greedy_decode_score = float(1.0 - sum(b.size for b in mblocks) / max(1, len(self.ref_words)))

        if greedy_decode_score > 0.5:
            self.aligner.log.info(
                f"poor score ({greedy_decode_score}): hyp_words={hyp_words[:30]}, ref_words={self.ref_words[:30]}")

        return LongAlignmentResult(
            spans,
            int(failed_words),
            greedy_decode_score,
            hyp_words=hyp_words,
            unspoken=[k in unspoken for k in range(len(self.ref_words))],
        )

    def _wildcards(self, indices: list[int]) -> list[int]:
        if not self.config.use_wildcards:
            return []

        return [
            k
            for k, word in enumerate(indices)
            if k == 0 or word in self.turn_starts or word != indices[k - 1] + 1
        ] + [len(indices)]

    def _choose_anchors(
            self,
            hyp_words: list[tuple[str, int, int]],
            n_frames: int,
            block_frames: int = 3000,
            blocks: list[difflib.Match] | None = None
    ) -> tuple[list[tuple[int, int]], set[int]]:
        """Finds a line of words (min_anchor_run words) that line up with CTC predictions as anchors for speech.

        A long enough run is trusted even when the transcript before it can't fit in the audio before it. Within every
        gap between anchors, transcript words that the greedy hypothesis doesn't support and that can't fit in the
        surrounding audio are returned as unspoken.
        """
        max_chars_per_frame = self.config.max_chars_per_frame
        min_anchor_run = self.config.min_anchor_run

        if blocks is None:
            blocks = self._match_blocks(hyp_words)

        candidates = []
        for h0, r0, size in blocks:
            if size < min_anchor_run:
                continue

            strong = size >= self.config.min_unspoken_anchor_run
            # Skip the first and last of the run; anchor on each interior word.
            for k in range(1, size - 1):
                head = [(r0 + j, hyp_words[h0 + j][1], hyp_words[h0 + j][2]) for j in range(k)]
                candidates.append((r0 + k, hyp_words[h0 + k][1], strong, head))

        # Prefix sum of characters so the feasibility test below is O(1) per candidate.
        cum = [0]
        for w in self.ref_words:
            cum.append(cum[-1] + len(w) + 1)

        anchors = []
        heads: list[list[tuple[int, int, int]]] = []
        last_frame = 0
        last_ref = 0
        total = cum[-1]
        for ref_i, frame, strong, head in candidates:
            if ref_i <= last_ref or frame <= last_frame:
                continue

            if frame - last_frame < block_frames and anchors:
                continue

            if cum[ref_i] - cum[last_ref] > (frame - last_frame) * max_chars_per_frame:
                if not strong or not self.config.detect_unspoken:
                    continue

                dropped = self._unspoken_in_gap(last_ref, ref_i, last_frame, frame, hyp_words, head)
                kept_chars = cum[ref_i] - cum[last_ref] - sum(len(self.ref_words[k]) + 1 for k in dropped)
                if kept_chars > (frame - last_frame) * max_chars_per_frame:
                    continue

            # the text still to come must also fit in the frames still to come
            elif not strong and total - cum[ref_i] > (n_frames - frame) * max_chars_per_frame:
                continue

            anchors.append((ref_i, frame))
            heads.append(head)
            last_frame = frame
            last_ref = ref_i

        unspoken: set[int] = set()
        if not self.config.detect_unspoken:
            return anchors, unspoken

        bounds = [(0, 0)] + anchors + [(len(self.ref_words), n_frames)]
        for i, ((r0, f0), (r1, f1)) in enumerate(itertools.pairwise(bounds)):
            head = heads[i] if i < len(heads) else []
            unspoken |= self._unspoken_in_gap(r0, r1, f0, f1, hyp_words, head)

        return anchors, unspoken

    def _unspoken_in_gap(
            self,
            r0: int,
            r1: int,
            f0: int,
            f1: int,
            hyp_words: list[tuple[str, int, int]],
            supported: list[tuple[int, int, int]] | None = None,
    ) -> set[int]:
        """Transcript words in [r0, r1) too long for the audio around them, given the hypothesis words that match.

        Stretches of transcript between hypothesis matches keep as many words as fit their audio at the maximum
        speaking rate, nearest the matches; the rest are unspoken. Nothing is dropped when there's no match to go by
        or when more than ``max_unspoken_words`` would be.
        """
        if supported is None:
            supported = []

        gap_hyp = [(word, start, end) for word, start, end in hyp_words if f0 <= start and end <= f1]
        points = {
            ref: (start, end)
            for ref, start, end in supported
            if r0 <= ref < r1
        }

        for a, b, size in self._consistent_runs(gap_hyp, r0, r1):
            for k in range(size):
                points.setdefault(b + k, (gap_hyp[a + k][1], gap_hyp[a + k][2]))

        if not points:
            return set()

        dropped: set[int] = set()
        ordered = sorted(points.items())
        stops = [(r0 - 1, (f0, f0))] + ordered + [(r1, (f1, f1))]
        for i, ((prev_ref, (_, prev_end)), (next_ref, (next_start, _))) in enumerate(itertools.pairwise(stops)):
            stretch = list(range(prev_ref + 1, next_ref))
            budget = (next_start - prev_end) * self.config.max_chars_per_frame
            if not stretch or self._chars(prev_ref + 1, next_ref) <= budget:
                continue

            if i == 0:
                kept = self._fitting(stretch[::-1], budget)
            elif i == len(stops) - 2:
                kept = self._fitting(stretch, budget)
            else:
                kept = self._fitting(stretch, budget / 2) | self._fitting(stretch[::-1], budget / 2)

            dropped.update(k for k in stretch if k not in kept)

        if len(dropped) > self.config.max_unspoken_words:
            return set()

        return dropped

    def _consistent_runs(self, gap_hyp: list[tuple[str, int, int]], r0: int, r1: int) -> list[tuple[int, int, int]]:
        """In-order (hyp index, ref index, size) runs covering the most words with time-feasible text between them."""
        positions: dict[str, list[int]] = {}
        for b in range(r0, r1):
            positions.setdefault(self.ref_words[b], []).append(b)

        runs = []
        for a, (word, _, _) in enumerate(gap_hyp):
            for b in positions.get(word, []):
                if a > 0 and b > r0 and gap_hyp[a - 1][0] == self.ref_words[b - 1]:
                    continue

                size = 1
                while a + size < len(gap_hyp) and b + size < r1 and gap_hyp[a + size][0] == self.ref_words[b + size]:
                    size += 1

                if size >= self.config.min_unspoken_support_run:
                    runs.append((a, b, size))

        runs.sort(key=lambda run: (run[1], run[0]))
        best = [run[2] for run in runs]
        previous: list[int | None] = [None] * len(runs)
        for j, (a_j, b_j, size_j) in enumerate(runs):
            for i in range(j):
                a_i, b_i, size_i = runs[i]
                if a_i + size_i > a_j or b_i + size_i > b_j:
                    continue

                frames = gap_hyp[a_j][1] - gap_hyp[a_i + size_i - 1][2]
                if self._chars(b_i + size_i, b_j) > frames * self.config.max_chars_per_frame:
                    continue

                if best[i] + size_j > best[j]:
                    best[j] = best[i] + size_j
                    previous[j] = i

        if not runs:
            return []

        chain = []
        j: int | None = max(range(len(runs)), key=lambda k: best[k])
        while j is not None:
            chain.append(runs[j])
            j = previous[j]

        return chain[::-1]

    def _chars(self, start: int, end: int) -> int:
        return int(self.cum_chars[end] - self.cum_chars[start])

    def _fitting(self, words: list[int], budget: float) -> set[int]:
        kept = set()
        used = 0
        for k in words:
            used += len(self.ref_words[k]) + 1
            if used > budget:
                break
            kept.add(k)
        return kept

    def _split_long(self, blocks: list[AlignmentBlock], max_block_frames: int) -> list[AlignmentBlock]:
        """Using boundaries from alignment calculations, splits long boundaries into smaller plausible sections."""
        out = []
        for indices, f0, f1 in blocks:
            n_frames = f1 - f0
            if n_frames <= max_block_frames or len(indices) < 2:
                out.append((indices, f0, f1))
                continue

            parts = int(np.ceil(n_frames / max_block_frames))
            lens = np.array([len(self.ref_words[k]) + 1 for k in indices], dtype=np.float64)
            cum = np.concatenate([[0.0], np.cumsum(lens)])

            start = 0
            frame = f0
            for p in range(parts):
                if p == parts - 1:
                    end = len(indices)
                    frame_end = f1
                else:
                    target = cum[-1] * (p + 1) / parts
                    end = min(max(int(np.searchsorted(cum, target)), start + 1), len(indices))
                    frame_end = f0 + int(round(n_frames * (p + 1) / parts))

                if end > start:
                    out.append((indices[start:end], frame, frame_end))

                start, frame = end, frame_end
                if start >= len(indices):
                    break

        return out

    def _match_blocks(self, hyp_words: list[tuple[str, int, int]]) -> list[difflib.Match]:
        """Matches strings together."""
        hyp = [w for w, _, _ in hyp_words]
        return difflib.SequenceMatcher(None, hyp, self.ref_words, autojunk=False).get_matching_blocks()


class TokenTimeline:
    """Helper/utility class for the finalized spans for segments from a long speech example."""

    spans: list[SpanScore | None]
    ok_mask: list[bool]

    def __init__(self, times: list[SpanScore | None], groups: list[int]) -> None:
        """Simple init."""
        self.times = times
        self.groups = groups

    def fill_gaps(self) -> None:
        """Algorithm for attempting to extend spans to meet each other with a maximum allowed amount of overlap."""
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

                    pos = bisect.bisect_left(idx, k)
                    prev = idx[pos - 1] if pos > 0 else None
                    nxt = idx[pos] if pos < len(idx) else None

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
                        raise ValueError("could not determine start and end for filled gap")
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
