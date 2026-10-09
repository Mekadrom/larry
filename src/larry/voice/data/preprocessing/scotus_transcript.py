import difflib
import re
from datetime import date

from larry.voice.config.preprocessing.voice_configs import SCOTUSTranscriptConfig
from larry.voice.data.preprocessing import transcript
from larry.voice.data.preprocessing.transcript import Transcript

# a speaker label with an explicit title prefix. the prefix is always all-caps in a label and mixed case in speech
# ("Mr.", "Justice"), which makes it safe to detect mid-line. surnames may contain non-ascii letters ("AGUIÑAGA"),
# apostrophes, hyphens, and lowercase letters ("McGRATH", "LaCOUR").
_LABEL = r"(?:CHIEF\s+JUSTICE|JUSTICE|GENERAL|MRS?\.|MS\.)\s+[A-Z](?:[^\W\d_]|['-])*[A-Z](?:[^\W\d_]|['-])*"

# match a bunch of raw spaces before the line number(s), followed by at least 2 tabs or spaces, followed by any
# single non-whitespace character and any character thereafter, then any whitespace until the end of the line.
_BODY = re.compile(r"^ {0,10}(\d{1,2})[ \t]{2,}(\S.*?)\s*$")
# split point in front of a titled label that the reporter didn't put on its own line; the terminator may be a colon
# or (as a reporter typo) a period. the lookbehind stops "CHIEF JUSTICE ROBERTS" from being split at "JUSTICE"
_MIDLINE = re.compile(rf"(?<!CHIEF)\s(?={_LABEL}(?:[:.]+(?:\s|$)|\s+--))")

# marks the start of dialog
_PROCEEDINGS = re.compile(r"^P\s?R\s?O\s?C\s?E\s?E\s?D\s?I\s?N\s?G\s?S\s*[.:;,]?\s*$")
# marks the end of dialog
_WHEREUPON = re.compile(r"^(?:\(\s*Whereupon\b|Whereupon\b.*\d{1,2}:\d{2}\s*[ap]\.?\s?m)", re.IGNORECASE)
# as a ripcord fallback, try to match the first word index entry line to end dialog. segmenting can clean up whatever
# garbage gets caught if this does happen
_INDEX_LINE = re.compile(r"^[A-Za-z][A-Za-z'\u2019-]*\s+\[\d+\]\s+\d")

# any line which consists of bracketed text only, including potential punctuation. captures stage directions like
# "(Laughter)" and variants like "(Laughter.)" or typos thereof, as well as plain timestamps like "(11:39 a.m.)" or
# typos thereof (unmatched brackets notable) - there is an edge case where a line of real dialog can start with a
# parenthetical and end with a different parenthetical, so this also makes sure all the content between the opening and
# closing parenthesis are not themselves parentheses.
_NOTE_CLOSED = re.compile(r"^[(\[{]([^(){}\[\]]+)[}\])](\.)*?$")

# stage directions and inserted notes can occasionally be unbounded or across multiple lines)
_NOTE_OPEN = re.compile(r"^\([A-Z][^)]*$")

# matches the stuff before the actual transcript in the pdf and the "P R O C E E D I N G S" which starts the dialog
_HEADING = re.compile(
    r"^(?:(?:ORAL|REBUTTAL|REDIRECT|CONTINUED|FURTHER)\s+)*ARGUMENT\s+(?:OF|BY)\b"
    r"|^ON\s+BEHALF\s+OF\b"
    r"|^P\s?R\s?O\s?C\s?E\s?E\s?D\s?I\s?N\s?G\s?S\s*[.:;,]?\s*$"
    r"|^C\s?E\s?R\s?T\s?I\s?F\s?I\s?C\s?A\s?T\s?E\s*[.:;,]?\s*$"
)

# a titled label at the start of a line, terminated by a colon or a period
_TITLED = re.compile(rf"^({_LABEL})(?:[:.]+|(?=\s+--))(?:\s+(.*))?$")
# fallback for labels without a title prefix ("THE COURT", "UNIDENTIFIED", bare surnames): requires that they start
# with a capital letter first. then they can consist of any combination of non-nonword and non-numeric characters,
# period (eg "MR.", "MRS.", apostrophes ("O'LEARY"), ampersands, slashes, spaces, and hyphens, a literal ":" colon,
# and then any trailing content is captured (speech)
_SPEAKER = re.compile(r"^(?!RCRA:)([A-Z](?:[^\W\d_]|[.'&/ -]){1,43}):(?:\s+(.*))?$")

_APPEARANCES = re.compile(r"^APPEARANCES\s*:?\s*$", re.IGNORECASE)
_APPEARANCE_NAME = re.compile(r"^((?:[^\W\d_]|[.'\- ]){4,60}?(?:,\s*(?:JR|SR|I{1,3}|IV)\.?)?),\s")

_BENCH = [
    ("CHIEF JUSTICE ROBERTS", date(2005, 9, 29), None),
    ("JUSTICE SCALIA", date(1986, 9, 26), date(2016, 2, 13)),
    ("JUSTICE KENNEDY", date(1988, 2, 18), date(2018, 7, 31)),
    ("JUSTICE THOMAS", date(1991, 10, 23), None),
    ("JUSTICE GINSBURG", date(1993, 8, 10), date(2020, 9, 18)),
    ("JUSTICE BREYER", date(1994, 8, 3), date(2022, 6, 30)),
    ("JUSTICE ALITO", date(2006, 1, 31), None),
    ("JUSTICE SOTOMAYOR", date(2009, 8, 8), None),
    ("JUSTICE KAGAN", date(2010, 8, 7), None),
    ("JUSTICE GORSUCH", date(2017, 4, 10), None),
    ("JUSTICE KAVANAUGH", date(2018, 10, 6), None),
    ("JUSTICE BARRETT", date(2020, 10, 27), None),
    ("JUSTICE JACKSON", date(2022, 6, 30), None),
]
_BENCH_TITLE = re.compile(r"^(?:THE\s+)?(?:CHIEF\s+)?(?:JUSTICE|JUSTICIE|JSUTICE|JUST|JUDGE)\b")

# matches titles/epithets
_JUSTICES = "JUSTICE|CHIEF JUSTICE|THE CHIEF JUSTICE"
HONORIFIC = re.compile(rf"^(MR|MS|MRS|DR|GENERAL|{_JUSTICES})\.?\s+", re.IGNORECASE)

NAME_SUFFIX = re.compile(r"^(JR|SR|I{1,3}|IV|V|ESQ)\.?$", re.IGNORECASE)

# reporter typos
_ALIASES = {
    "ERICJ FEIGIN": "ERIC FEIGIN",
    "JEFFERY FISHER": "JEFFREY FISHER",
    "IRV GORNSTEIN": "IRVING GORNSTEIN",
    "STUART DUNCAN": "KYLE DUNCAN",
}


class SCOTUSTranscript(Transcript[SCOTUSTranscriptConfig]):
    def __init__(self, config: SCOTUSTranscriptConfig, docket: str, date_argued: date) -> None:
        super().__init__(config)
        self.docket = docket
        self.date_argued = date_argued
        self.bench = {
            name.split()[-1]: name
            for name, a, b in _BENCH
            if a <= self.date_argued and (b is None or self.date_argued <= b)
        }

    def extract_body_lines(self, raw: str) -> list[str]:
        out = []
        for line in raw.splitlines():
            # replace weird characters with regular ones
            line = (line.replace("\f", "").replace("\u00ad", "-")
                    .replace("\u2019", "'").replace("\u2018", "'")
                    .replace("\u201c", '"').replace("\u201d", '"')
                    .replace("\u2014", "--").replace("\u2013-", "--")
                    .replace("-\u2013", "--").replace("\u2013", "--")
                    .replace("\u2010", "-").replace("\u2011", "-")
                    .replace("\u2212", "-").replace("\u2015", "-"))

            # a lone spaced single hyphen (ascii or former soft hyphen) is an interruption; make it "--" to be consistent
            line = re.sub(r"(?:(?<=\s)|^)-(?=\s|$)", "--", line)
            line = re.sub(r"-{3,}", "--", line)

            # match only lines with line number and specific whitespace formatting
            m = _BODY.match(line)
            if not m:
                continue

            # sanity check line number (not many per page in gt transcripts)
            n = int(m.group(1))
            if not 1 <= n <= 30:
                continue
            # one label per line from here on: split in front of any label embedded mid-line
            out.extend(p.strip() for p in _MIDLINE.split(m.group(2)) if p.strip())
        return out

    def parse_turns(self) -> None:
        start, end = self._parse_start_end(self.body_lines)

        speaker = None
        speech_buffer = []
        buf_line = start
        paren = None

        for i in range(start + 1, end):
            text = self.body_lines[i]

            # multi-line notes
            # only known valid case is a mid-argument recess (19-368)
            if paren is not None:
                paren.append(text)
                if ")" in text:
                    paren = None
                    continue

                if len(paren) < self.config.max_note_lines:
                    continue
                # not a note after all; it was speech. the current line is processed below, so leave it out here
                speech_buffer.extend(paren[:-1])
                paren = None

            if self._looks_like_note_text(text) is not None:
                continue

            if _NOTE_OPEN.match(text) and " " in text:
                # the note is skipped without ending the turn; the current speaker continues after it
                paren = [text]
                continue

            if self._looks_like_heading(text):
                continue

            # check if we're starting a new speaker dialog
            speaker_label_match = self._match_title_or_speaker_label(text)
            if speaker_label_match:
                # new speaker's turn, flush the previous speaker's turn
                self.flush_turn(speaker, speech_buffer, buf_line)
                speech_buffer = []

                # get the real speaker label without extra whitespace
                speaker = re.sub(r"\s+", " ", speaker_label_match.group(1)).strip()

                # whatever remains after the part that actually consists of the speaker's label is considered speech
                rest = (speaker_label_match.group(2) or "").strip()
                if rest:
                    speech_buffer.append(rest)

                buf_line = i
                continue

            if speaker is None:
                speaker = "UNIDENTIFIED"
                buf_line = i

            speech_buffer.append(text)

        self.flush_turn(speaker, speech_buffer, buf_line)
        self.load_appearances()

    def load_appearances(self) -> None:
        started = False
        for ln in self.body_lines:
            if _APPEARANCES.match(ln):
                started = True
                continue

            if not started:
                continue

            if _PROCEEDINGS.match(ln):
                break

            m = _APPEARANCE_NAME.match(ln)
            if m and self._is_label(m.group(1), upper_ratio=0.8):
                full = re.sub(r"\s+", " ", m.group(1)).strip()
                for key in self._name_keys(full):
                    self.appearances.setdefault(key, full)

    @staticmethod
    def _name_keys(full: str) -> set[str]:
        toks = [
            t
            for t in full.upper().replace(",", " ").split()
            if t.strip(".")
        ]
        while toks and NAME_SUFFIX.match(toks[-1]):
            toks.pop()
        return {" ".join(toks[-k:]) for k in (1, 2, 3) if len(toks) >= k}

    @staticmethod
    def _parse_start_end(lines: list[str]) -> tuple[int, int]:
        start = None
        end = None
        for i, line in enumerate(lines):
            if _PROCEEDINGS.match(line):
                start = i
            if start is not None and i > start and _WHEREUPON.match(line):
                end = i

        if start is None:
            raise ValueError("no 'P R O C E E D I N G S' marker; transcript layout not recognized")

        if end is None:
            end = len(lines)
            for i, line in enumerate(lines[start:], start):
                if _INDEX_LINE.match(line):
                    end = i
                    break

        return start, end

    @staticmethod
    def _looks_like_note_text(text):
        # match opening and closing parentheses appearing uninterrupted in one line
        m = _NOTE_CLOSED.match(text)
        if not m:
            return None

        inner = m.group(1)
        if not any(c.islower() for c in inner):
            return None

        if " " not in inner and "." not in inner:
            return None

        return text

    @staticmethod
    def _looks_like_heading(text: str) -> bool:
        if _HEADING.search(text):
            return True

        # headers are all-caps with no colon and more than one word
        # speaker lines always have a colon and speaker continuation lines always have lowercase letters somewhere
        if ":" in text:
            return False

        if any(c.islower() for c in text):
            return False

        letters = [
            c
            for c in text
            if c.isalpha()
        ]
        return len(letters) >= 4 and len(text.split()) >= 2

    def _match_title_or_speaker_label(self, text: str) -> re.Match | None:
        match = _TITLED.match(text)
        if match is None:
            match = _SPEAKER.match(text)
            if match:
                is_label = self._is_label(match.group(1))
                is_heading = _HEADING.search(match.group(1))

                if not is_label or is_heading:
                    match = None

        return match

    @staticmethod
    def _is_label(label: str, upper_ratio: float = 0.6) -> bool:
        letters = [
            c
            for c in label
            if c.isalpha()
        ]

        if len(letters) < 2:
            return False

        return sum(c.isupper() for c in letters) / len(letters) >= upper_ratio

    def flush_turn(self, speaker: str | None, buffer: list[str], buffer_line: int) -> None:
        if speaker is not None and buffer:
            # normalize whitespace
            text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
            # repair a trailing double hyphen that got cut down to a single one
            text = re.sub(r"(?<=\s)-$", "--", text)
            text = re.sub(r"(?<=[^\s-])-$", " --", text)
            if text:
                self.turns.append(transcript.Turn(speaker=speaker, text=text, line_start=buffer_line))

    def resolve_speaker(self, speaker_raw: str) -> tuple[str, str]:
        up = speaker_raw.strip().rstrip(":").strip().upper()
        if not up:
            return f"scotus:{self.docket}:UNIDENTIFIED", "unknown"

        if up in ("CHIEF JUSTICE", "THE CHIEF JUSTICE"):
            chief = next((v for v in self.bench.values() if v.startswith("CHIEF")), None)
            if chief:
                return f"scotus:{chief}", "justice"

        if up in ("UNIDENTIFIED", "THE COURT"):
            return f"scotus:{self.docket}:{up}", "unknown"

        surname = up.split()[-1].strip(".,")
        if _BENCH_TITLE.match(up):
            near = difflib.get_close_matches(surname, list(self.bench), n=1, cutoff=0.75)
            if near:
                return f"scotus:{self.bench[near[0]]}", "justice"
            return f"scotus:{self.docket}:{up}", "unknown"

        rest = HONORIFIC.sub("", up).strip(" .,")
        if rest in self.appearances:
            return f"scotus:adv:{self._adv_key(self.appearances[rest])}", "advocate"

        surname = rest.split()[-1] if rest else up
        close = difflib.get_close_matches(surname, list(self.appearances), n=1, cutoff=0.82)
        if close:
            return f"scotus:adv:{self._adv_key(self.appearances[close[0]])}", "advocate"

        if surname in self.bench:  # e.g. "MR. KAGAN", a reporter typo for the justice
            return f"scotus:{self.bench[surname]}", "justice"
        return f"scotus:{self.docket}:{up}", "advocate"

    @staticmethod
    def _adv_key(full: str) -> str:
        toks = [t for t in re.split(r"[\s.,]+", full.upper()) if t]
        while toks and toks[0] in {"GEN", "GENERAL", "HON", "MR", "MS", "MRS", "DR", "PROF"}:
            toks.pop(0)

        while toks and NAME_SUFFIX.match(toks[-1]):
            toks.pop()

        if len(toks) >= 3 and len(toks[0]) == 1:  # "E JOSHUA ROSENKRANZ", "D JOHN SAUER"
            toks = toks[1:]

        key = f"{toks[0]} {toks[-1]}" if len(toks) >= 2 else " ".join(toks)
        return _ALIASES.get(key, key)
