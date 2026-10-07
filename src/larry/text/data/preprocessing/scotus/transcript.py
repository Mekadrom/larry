import dataclasses
import os
import re
import shutil
import subprocess

_UPPER_RATIO = 0.6
_MAX_NOTE_LINES = 4


@dataclasses.dataclass
class Turn:
    speaker: str
    text: str
    line_start: int


@dataclasses.dataclass
class Transcript:
    turns: list[Turn] = dataclasses.field(default_factory=list)


def pdf_to_text(pdf_path: str) -> str:
    """`pdftotext -layout`, which preserves the line-number column the parser keys on."""
    exe = shutil.which("pdftotext")
    if exe is None:
        raise RuntimeError(
            "pdftotext not found. Install poppler-utils (apt install poppler-utils). It is "
            "used as a subprocess so the parser needs no new Python dependency."
        )
    txt_path = pdf_path + ".txt"
    if not (os.path.exists(txt_path) and os.path.getsize(txt_path) > 0):
        subprocess.run([exe, "-layout", "-enc", "UTF-8", pdf_path, txt_path], check=True, capture_output=True)
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# match a bunch of raw spaces before the line number(s), followed by at least 2 tabs or spaces, followed by any
# single non-whitespace character and any character thereafter, then any whitespace until the end of the line.
_BODY = re.compile(r"^ {0,10}(\d{1,2})[ \t]{2,}(\S.*?)\s*$")
_LABEL = r"(?:CHIEF JUSTICE|JUSTICE|GENERAL|MRS?\.|MS\.) [A-Z](?:[^\W\d_]|['-])+"
_MIDLINE = re.compile(rf"\s(?={_LABEL}[:.]\s)")
_TITLED = re.compile(rf"^({_LABEL})[:.]\s+(.*)$")

def _body_lines(raw: str) -> str:
    out = []
    for line in raw.splitlines():
        # replace weird characters with regular ones
        line = (line.replace("\f", "").replace("\u00ad", "-")
                .replace("\u2019", "'").replace("\u2018", "'")
                .replace("\u201c", '"').replace("\u201d", '"'))
        m = _BODY.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if not 1 <= n <= 30:
            continue
        out.extend(p.strip() for p in _MIDLINE.split(m.group(2)) if p.strip())
    return out


# any line which consists of bracketed text only, including potential punctuation. captures stage directions like
# "(Laughter)" and variants like "(Laughter.)" or typos thereof, as well as plain timestamps like "(11:39 a.m.)" or
# typos thereof (unmatched brackets notable) - there is an edge case where a line of real dialog can start with a
# parenthetical and end with a different parenthetical, so this also makes sure all the content between the opening and
# closing parenthesis are not themselves parentheses.
_NOTE_CLOSED = re.compile(r"^[(\[{]([^(){}\[\]]+)[}\])](\.)*?$")


def _note_text(text):
    m = _NOTE_CLOSED.match(text)
    if not m:
        return None
    inner = m.group(1)
    if not any(c.islower() for c in inner):
        return None
    if " " not in inner and "." not in inner:
        return None
    return text


# matches the stuff before the actual transcript in the pdf and the "P R O C E E D I N G S" which starts the dialog
_HEADING = re.compile(
    r"^(?:(?:ORAL|REBUTTAL|REDIRECT|CONTINUED|FURTHER)\s+)*ARGUMENT\s+(?:OF|BY)\b"
    r"|^ON\s+BEHALF\s+OF\b"
    r"|^P\s?R\s?O\s?C\s?E\s?E\s?D\s?I\s?N\s?G\s?S\s*[.:;,]?\s*$"
    r"|^C\s?E\s?R\s?T\s?I\s?F\s?I\s?C\s?A\s?T\s?E\s*[.:;,]?\s*$"
)


def _looks_like_heading(text):
    if _HEADING.search(text):
        return True
    # headers are all-caps with no colon and more than one word
    # speaker lines always have a colon and speaker continuation lines always have lowercase letters somewhere
    if ":" in text:
        return False
    if any(c.islower() for c in text):
        return False
    letters = [c for c in text if c.isalpha()]
    return len(letters) >= 4 and len(text.split()) >= 2


def _is_label(label):
    letters = [c for c in label if c.isalpha()]
    if len(letters) < 2:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= _UPPER_RATIO


# marks the start of dialog
_PROCEEDINGS = re.compile(r"^P\s?R\s?O\s?C\s?E\s?E\s?D\s?I\s?N\s?G\s?S\s*[.:;,]?\s*$")
# marks the end of dialog
_WHEREUPON = re.compile(r"^\(\s*Whereupon\b", re.I)
# marks the end of dialog (for the weird examples that don't match the previous regex)
_WHEREUPON_LOOSE = re.compile(r"^Whereupon\b.*\d{1,2}:\d{2}\s*[ap]\.?\s?m", re.I)
# as a ripcord fallback, try to match the first word index entry line to end dialog. segmenting can clean up whatever
# garbage gets caught if this does happen
_INDEX_LINE = re.compile(r"^[A-Za-z][A-Za-z'\u2019-]*\s+\[\d+\]\s+\d")
# stage directions and inserted notes can occasionally be unbounded or across multiple lines)
_NOTE_OPEN = re.compile(r"^\([A-Z][^)]*$")
# match speaker names, requiring that they start with a capital letter first. then they can consist of any combination
# of non-nonword and non-numeric characters, period (eg "MR.", "MRS.", apostrophes ("O'LEARY"), ampersands, slashes,
# spaces, and hyphens, a literal ":" colon, and then any trailing content is captured (speech)
_SPEAKER = re.compile(r"^(?!RCRA:)([A-Z](?:[^\W\d_]|[.'&/ -]){1,43}):(?:\s+(.*))?$")


def parse(url_hash: str) -> Transcript:
    raw = pdf_to_text(url_hash)
    lines = _body_lines(raw)

    start = None
    end = None
    for i, ln in enumerate(lines):
        if _PROCEEDINGS.match(ln):
            start = i
        if start is not None and i > start and (_WHEREUPON.match(ln) or _WHEREUPON_LOOSE.match(ln)):
            end = i

    if start is None:
        raise ValueError("no 'P R O C E E D I N G S' marker; transcript layout not recognized")

    if end is None:
        end = len(lines)
        for i, ln in enumerate(lines[start:]):
            if _INDEX_LINE.match(ln):
                end = i

    tr = Transcript()

    speaker = None
    buf = []
    buf_line = start
    paren = None

    def flush():
        nonlocal speaker, buf
        if speaker is not None and buf:
            # normalize whitespace
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            # repair a trailing hyphen that got cutoff because i0 can't find which other regex is removing them
            text = re.sub(r"(?<=\s)-$", "--", text)
            if text:
                tr.turns.append(Turn(speaker=speaker, text=text, line_start=buf_line))
        buf = []

    for i in range(start + 1, end):
        text = lines[i]

        # notes may wrap across numbered lines; consume until it closes, but give up
        # after a few rather than swallowing speech that never had a closing paren.
        if paren is not None:
            paren.append(text)
            if ")" in text:
                paren = None
                continue
            if len(paren) < _MAX_NOTE_LINES:
                continue
            # not a note after all; it was speech
            buf.extend(paren)
            paren = None

        note = _note_text(text)
        if note is not None:
            continue
        if _NOTE_OPEN.match(text) and " " in text:
            paren, paren_line = [text], i
            continue

        if _looks_like_heading(text):
            continue

        m = _SPEAKER.match(text)
        if m and _is_label(m.group(1)) and not _HEADING.search(m.group(1)):
            flush()
            speaker = re.sub(r"\s+", " ", m.group(1)).strip()
            buf_line = i
            rest = (m.group(2) or "").strip()
            if rest:
                buf.append(rest)
            continue

        if speaker is None:
            # Text before any speaker label inside the proceedings. Rare; keep it under a
            # sentinel rather than silently discarding audio-bearing words.
            speaker = "UNIDENTIFIED"
            buf_line = i
        buf.append(text)

    flush()
    return tr
