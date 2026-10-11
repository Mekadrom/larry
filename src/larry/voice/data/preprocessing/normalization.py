import re
import unicodedata

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
_ROMAN = re.compile(r"(X{0,3})(IX|IV|V?I{0,3})")
_ROMAN_HEAD = re.compile(r"^([IVX]+)(?![A-Za-z])(.*)$")
_ROMAN_TRIGGERS = {"ARTICLE", "TITLE", "WAR", "SCHEDULE", "PHASE"}
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}

_ABBREVIATIONS_NO_PERIOD = {"MR", "MRS", "V"}
ABBREVIATIONS: dict[str, list[str]] = {
    "MR": ["MISTER"],
    "MRS": ["MISSUS"],
    "MS": ["MIZ", "MISS"],
    "DR": ["DOCTOR", "DRIVE"],
    "ST": ["SAINT", "STREET"],
    "LT": ["LIEUTENANT"],
    "JR": ["JUNIOR"],
    "GEN": ["GENERAL"],
    "V": ["VERSUS", "V"],
    "VS": ["VERSUS", "V"],
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

    if 1100 <= n <= 9999:
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


def pairs(s: str) -> str:
    """Docket/section style: 316 -> THREE SIXTEEN, 7201 -> SEVENTY TWO OH ONE."""
    if len(s) == 3:
        head = _ONES[int(s[0])]
        lo = int(s[1:])
    elif len(s) == 4:
        head = cardinal(int(s[:2]))
        lo = int(s[2:])
    else:
        return cardinal(int(s))

    if lo == 0:
        return head + " HUNDRED"
    if lo < 10:
        return head + " OH " + _ONES[lo]
    return head + " " + cardinal(lo)


def roman_value(numeral: str) -> int:
    total = 0
    for i, symbol in enumerate(numeral):
        value = _ROMAN_VALUES[symbol]
        is_subtractive = i + 1 < len(numeral) and _ROMAN_VALUES[numeral[i + 1]] > value
        if is_subtractive:
            total -= value
        else:
            total += value
    return total


class TextNormalizer:
    def normalize_token(self, token: str, prev_token: str | None = None) -> list[list[str]]:
        """One original token -> zero or more CTC-alphabet words."""

        roman = self._roman_readings(token, prev_token)
        if roman is not None:
            return roman

        abbreviations = self._abbreviation_readings(token)
        if abbreviations is not None:
            return abbreviations

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
            return [(number_token + " DOLLARS").split()]

        for sym, rep in _SYMBOL.items():
            token = token.replace(sym, rep)

        # try the token whole before splitting it
        bare = token.strip("(){}[]\"'.,;:!?")
        readings = self._number_readings(bare)
        if len(readings) > 0:
            return [
                r.split()
                for r in readings
            ]

        piece_options: list[list[list[str]]] = []
        # split on anything that is non-alphanumeric;
        # makes "24-316" and "U.S.C." pronounceable
        for piece in re.split(r"[^A-Za-z0-9']+", token):
            if not piece:
                continue

            readings = self._number_readings(piece)
            if len(readings) > 0:
                piece_options.append([
                    r.split()
                    for r in readings
                ])
                continue

            if any(c.isdigit() for c in piece):
                # mixed like "1983a" or "401k": read the digit runs out, keep letter runs.
                words = []
                for run in re.findall(r"\d+|[A-Za-z']+", piece):
                    if run[0].isdigit():
                        spoken = self.number_token(run)
                        if spoken is None:
                            spoken = digits(run)
                        words.extend(spoken.split())
                    else:
                        words.append(run.upper())
                piece_options.append([words])
                continue

            word = "".join(c for c in piece.upper() if c in VOCAB_CHARS).strip("'")
            if word:
                piece_options.append([[word]])

        return self._combine(piece_options)

    def _abbreviation_readings(self, token: str) -> list[list[str]] | None:
        core = token.lstrip("(\"'").rstrip(",;:!?)\"'")
        has_period = core.endswith(".")
        key = core.rstrip(".").upper()
        if key not in ABBREVIATIONS:
            return None
        if not has_period and key not in _ABBREVIATIONS_NO_PERIOD:
            return None
        return [
            [word]
            for word in ABBREVIATIONS[key]
        ]

    def _roman_readings(self, token: str, prev_token: str | None) -> list[list[str]] | None:
        if prev_token is None:
            return None
        core = token.strip(".,;:!?()\"'")
        m = _ROMAN_HEAD.match(core)
        if m is None or _ROMAN.fullmatch(m.group(1)) is None:
            return None
        numeral = m.group(1)
        rest = m.group(2)
        prev = prev_token.strip(".,;:!?()\"'")
        value = roman_value(numeral)
        if prev.upper() in _ROMAN_TRIGGERS:
            readings = [cardinal(value).split()]
        elif len(numeral) >= 2 and prev[:1].isupper():
            readings = [
                cardinal(value).split(),
                ["THE"] + ordinal(value).split(),
            ]
        else:
            return None

        if rest.upper() == "'S":
            for reading in readings:
                reading[-1] += "'S"
            return readings

        if rest == "":
            return readings
        return self._combine([readings, self.normalize_token(rest)])

    def _combine(self, piece_options: list[list[list[str]]]) -> list[list[str]]:
        ambiguous = [
            options
            for options in piece_options
            if len(options) > 1
        ]
        if len(ambiguous) > 1:
            # several ambiguous numbers in one token: too many combinations, keep defaults
            piece_options = [
                options[:1]
                for options in piece_options
            ]

        readings: list[list[str]] = [[]]
        for options in piece_options:
            readings = [
                done + option
                for done in readings
                for option in options
            ]
        return readings

    def _number_readings(self, token: str) -> list[str]:
        """Candidate spoken forms of a number token, most likely first; empty if not a number."""
        default = self.number_token(token)
        if default is None:
            return []

        readings = [default]
        t = token.replace(",", "")
        is_year = re.fullmatch(r"\d{4}", t) is not None and 1100 <= int(t) <= 2099
        if re.fullmatch(r"\d{3,4}", t) is not None and not is_year:
            candidates = [pairs(t), digits(t)]
            if len(t) == 4 and int(t[:2]) % 10 != 0:
                lo = int(t[2:])
                hundreds = cardinal(int(t[:2])) + " HUNDRED"
                if lo > 0:
                    hundreds = hundreds + " " + cardinal(lo)
                candidates.append(hundreds)
            for candidate in candidates:
                if candidate not in readings:
                    readings.append(candidate)
        return readings

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

        # statutes; looks very similar to year
        t = token.replace(",", "")
        if re.fullmatch(r"\d{4}", t) and 2100 <= int(t) <= 9999:
            return year(int(t))

        return super().number_token(token)
