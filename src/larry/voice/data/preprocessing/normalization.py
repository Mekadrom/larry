import re
import unicodedata
from abc import ABC

VOCAB_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ'")

_ONES = [
    "ZERO",
    "ONE",
    "TWO",
    "THREE",
    "FOUR",
    "FIVE",
    "SIX",
    "SEVEN",
    "EIGHT",
    "NINE",
    "TEN",
    "ELEVEN",
    "TWELVE",
    "THIRTEEN",
    "FOURTEEN",
    "FIFTEEN",
    "SIXTEEN",
    "SEVENTEEN",
    "EIGHTEEN",
    "NINETEEN"
]
_TENS = ["", "", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY"]
_SCALES = [(1_000_000_000, "BILLION"), (1_000_000, "MILLION"), (1_000, "THOUSAND"), (100, "HUNDRED")]
_ORDINAL = {
    "ONE": "FIRST",
    "TWO": "SECOND",
    "THREE": "THIRD",
    "FIVE": "FIFTH",
    "EIGHT": "EIGHTH",
    "NINE": "NINTH",
    "TWELVE": "TWELFTH",
    "TWENTY": "TWENTIETH",
    "THIRTY": "THIRTIETH",
    "FORTY": "FORTIETH",
    "FIFTY": "FIFTIETH",
    "SIXTY": "SIXTIETH",
    "SEVENTY": "SEVENTIETH",
    "EIGHTY": "EIGHTIETH", "NINETY": "NINETIETH"
}
_SYMBOL = {
    "%": " PERCENT ",
    "§§": " SECTIONS ",
    "§": " SECTION ",
    "&": " AND ",
    "+": " PLUS ",
    "=": " EQUALS ",
    "°": " DEGREES ",
}


def cardinal(n):
    if n < 0:
        return "MINUS " + cardinal(-n)

    if n < 20:
        return _ONES[n]

    if n < 100:
        t, r = divmod(n, 10)
        return _TENS[t] + (" " + _ONES[r] if r else "")

    for value, name in _SCALES:
        if n >= value:
            head, rest = divmod(n, value)
            out = cardinal(head) + " " + name
            return out + (" " + cardinal(rest) if rest else "")

    return _ONES[n]


def ordinal(n):
    words = cardinal(n).split()
    last = words[-1]
    if last in _ORDINAL:
        words[-1] = _ORDINAL[last]
    else:
        words[-1] = last + "TH"
    return " ".join(words)


def year(n):
    """Read 1983 as NINETEEN EIGHTY THREE, 2007 as TWO THOUSAND SEVEN."""
    if 2000 <= n <= 2009:
        return "TWO THOUSAND" + ("" if n == 2000 else " " + _ONES[n - 2000])

    if 1100 <= n <= 2099:
        hi, lo = divmod(n, 100)
        if lo == 0:
            return cardinal(hi) + " HUNDRED"

        if lo < 10:
            return cardinal(hi) + " OH " + _ONES[lo]

        return cardinal(hi) + " " + cardinal(lo)

    return cardinal(n)


def digits(s):
    return " ".join([
        _ONES[int(c)]
        for c in s
        if c.isdigit()
    ])


def pairs(s):
    """Docket/section style: 316 -> THREE SIXTEEN, 1199 -> ELEVEN NINETY NINE."""
    if len(s) == 3:
        return _ONES[int(s[0])] + " " + cardinal(int(s[1:]))

    if len(s) == 4:
        return cardinal(int(s[:2])) + " " + cardinal(int(s[2:]))

    return cardinal(int(s))


class TextNormalizer(ABC):
    def normalize_stream(self, tokens: list[str]) -> tuple[list[str], list[int]]:
        """(normalized words, owner index into `tokens` for each normalized word)."""
        words = []
        owner = []
        for i, tok in enumerate(tokens):
            for w in self.normalize_token(tok):
                words.append(w)
                owner.append(i)
        return words, owner

    def normalize_token(self, token: str) -> list[str]:
        """One original token -> zero or more CTC-alphabet words."""
        token = unicodedata.normalize("NFKD", token)
        token = "".join([
            c
            for c in token
            if not unicodedata.combining(c)
        ])
        token = token.replace("’", "'").replace("‘", "'")
        token = token.replace("—", " ").replace("–", " ").replace("‐", "-")

        money = token.lstrip("$").rstrip(".,;:!?)\"'")

        number_token = self.number_token(money)
        if token.lstrip("(\"'").startswith("$") and number_token:
            return (number_token + " DOLLARS").split()

        for sym, rep in _SYMBOL.items():
            token = token.replace(sym, rep)

        # try the token whole before splitting it
        bare = token.strip("(){}[]\"'.,;:!?")
        spoken = self.number_token(bare)
        if spoken is not None:
            return spoken.split()

        out = []
        # split on anything that is non-alphanumeric;
        # makes "24-316" and "U.S.C." pronounceable
        for piece in re.split(r"[^A-Za-z0-9']+", token):
            if not piece:
                continue

            spoken = self.number_token(piece)
            if spoken is not None:
                out.extend(spoken.split())
                continue

            if any(c.isdigit() for c in piece):
                # mixed like "1983a" or "401k": read the digit runs out, keep letter runs.
                for run in re.findall(r"\d+|[A-Za-z']+", piece):
                    spoken = self.number_token(run)
                    out.extend((spoken or digits(run) if run[0].isdigit() else run.upper()).split())
                continue

            word = "".join(c for c in piece.upper() if c in VOCAB_CHARS).strip("'")
            if word:
                out.append(word)

        return out

    def number_token(self, token: str) -> str | None:
        t = token.replace(",", "")
        m = re.fullmatch(r"(\d+)(st|nd|rd|th|ST|ND|RD|TH)", t)
        if m:
            return ordinal(int(m.group(1)))

        if re.fullmatch(r"\d{4}", t) and 1100 <= int(t) <= 2099:
            return year(int(t))

        if re.fullmatch(r"\d+", t):
            n = int(t)
            return cardinal(n) if n < 1_000_000_000_000 else digits(t)

        m = re.fullmatch(r"(\d+)\.(\d+)", t)
        if m:
            return cardinal(int(m.group(1))) + " POINT " + digits(m.group(2))

        return None


class SCOTUSTextNormalizer(TextNormalizer):
    def number_token(self, token: str) -> str | None:
        m = re.fullmatch(r"(\d{1,2})-(\d{3,4})", token.replace(",", ""))
        if m:
            return cardinal(int(m.group(1))) + " " + pairs(m.group(2))

        return super().number_token(token)
