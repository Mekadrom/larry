import dataclasses
import re

from larry.text.config.preprocessing.text_mapper_configs import VisualDescriptionExtractingMapperConfig

COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "violet", "pink", "brown", "black", "white", "gray", "grey",
    "gold", "golden", "silver", "bronze", "copper", "teal", "turquoise", "ivory", "cream", "beige", "tan", "crimson",
    "scarlet", "maroon", "indigo", "ochre", "amber", "cobalt", "emerald", "olive", "sepia", "monochrome", "colorful",
    "colored", "coloured", "multicolored", "dark", "light", "pale", "bright", "vivid", "muted", "metallic",
    "translucent", "transparent", "opaque", "shiny", "glossy", "matte", "color", "colors", "palette", "hue", "tones",
}
MATERIALS = {
    "paper", "wood", "wooden", "metal", "iron", "steel", "brass", "tin", "glass", "ceramic", "porcelain",
    "earthenware", "stoneware", "clay", "terracotta", "stone", "marble", "granite", "leather", "cotton", "silk",
    "wool", "linen", "velvet", "lace", "satin", "fabric", "textile", "cloth", "canvas", "oil", "watercolor", "ink",
    "pencil", "graphite", "charcoal", "gouache", "tempera", "enamel", "lacquer", "bone", "shell", "plastic", "rubber",
    "cardboard", "feather", "feathers", "beads", "beaded", "gilt", "gilded", "glaze", "glazed", "plaster", "bamboo",
    "straw", "wire", "vellum", "parchment", "albumen", "gelatin", "photograph", "print", "lithograph", "etching",
    "engraving", "woodcut", "drawing", "sketch", "painting",
}
DECORATION = {
    "painted", "carved", "engraved", "etched", "embossed", "printed", "woven", "embroidered", "stitched", "quilted",
    "appliqued", "inlaid", "incised", "stamped", "molded", "moulded", "cast", "blown", "folded", "pierced",
    "decorated", "decoration", "decorative", "ornament", "ornamented", "ornate", "motif", "motifs", "pattern",
    "patterns", "patterned", "design", "designs", "border", "borders", "stripe", "stripes", "striped", "floral",
    "flowers", "flower", "blossoms", "foliage", "leaves", "leaf", "vine", "vines", "geometric", "abstract", "scroll",
    "scrolls", "medallion", "lettering", "inscribed", "inscription", "letters", "numerals", "embellished",
    "embellishments", "trim", "fringe", "tassel", "tassels", "dots", "checkered", "plaid", "repeating", "crisp",
}
SHAPES = {
    "round", "rounded", "circular", "oval", "square", "rectangular", "triangular", "cylindrical", "conical",
    "spherical", "curved", "straight", "flat", "tall", "narrow", "wide", "shaped", "shape", "shapes", "rim", "base",
    "handle", "handles", "lid", "spout", "neck", "edge", "edges", "corner", "corners", "surface", "frame", "framed",
    "mat", "panel", "panels", "folds", "sleeves", "collar", "brim", "blade", "dial", "face",
}
COMPOSITION = {
    "left", "right", "center", "centre", "centered", "foreground", "background", "top", "bottom", "upper", "lower",
    "front", "middle", "above", "below", "beneath", "behind", "beside", "across", "viewer", "depicted", "depicts",
    "depicting", "shows", "showing", "shown", "features", "featuring", "portrays", "portrayed", "image", "scene",
    "composition", "figure", "figures", "seated", "standing", "sitting", "wearing", "wears", "holds", "holding",
    "posed", "profile", "facing", "gaze", "presented", "arranged", "characterizes", "appears", "surrounded",
}
APPEARANCE = {
    "face", "faces", "eyes", "eye", "hair", "cheeks", "lips", "mouth", "nose", "hands", "hand", "arms", "arm", "legs",
    "beard", "mustache", "moustache", "smile", "smiling", "expression", "dress", "gown", "coat", "hat", "cap",
    "jacket", "shirt", "skirt", "suit", "uniform", "tie", "bow", "scarf", "shawl", "apron", "boots", "shoes",
    "jewelry", "necklace", "earrings", "gloves", "veil", "robe", "costume", "portrait", "bust", "torso", "nude",
}
OBJECTS = {
    "vessel", "jar", "jug", "bottle", "bottles", "vial", "vials", "stopper", "stoppers", "box", "case", "plate",
    "bowl", "cup", "saucer", "teapot", "pitcher", "vase", "chair", "table", "envelope", "envelopes", "stamp",
    "stamps", "sheet", "card", "poster", "label", "tag", "sign", "button", "buttons", "badge", "pin", "medal",
    "coin", "key", "lamp", "clock", "watch", "mirror", "basket", "doll", "toy", "flag", "banner", "book", "cover",
    "page", "pages", "map", "chart", "diagram", "model", "sculpture", "statue", "figurine", "textile", "rug",
    "blanket", "quilt", "wallpaper", "tile", "tiles", "window", "door", "wall", "walls", "floor", "ceiling", "room",
    "interior", "exterior", "view", "landscape", "building", "street", "tree", "trees", "sky", "water", "river",
    "mountains", "horse", "bird", "birds", "dog", "cat", "fish", "animal", "animals", "containers", "container",
    "engine", "wheel", "wheels", "propeller", "wing", "wings", "cylinders", "markings", "scale", "needle", "bubble",
}
PRESENT = {"is", "are", "has", "have", "sits", "stands", "hangs", "rests", "covers", "surrounds", "unify", "unifies"}
HISTORY = {
    "born", "died", "death", "life", "lived", "career", "war", "wars", "moved", "moving", "founded", "established",
    "founder", "served", "worked", "studied", "taught", "married", "family", "father", "mother", "son", "daughter",
    "wife", "husband", "brother", "sister", "president", "government", "congress", "collection", "collector",
    "donated", "donor", "gift", "acquired", "purchased", "commissioned", "exhibited", "exhibition", "museum",
    "history", "historical", "historian", "according", "believed", "known", "famous", "popular", "legacy",
    "community", "movement", "company", "firm", "business", "manufacturer", "patented", "patent", "invented",
    "inventor", "sold", "owned", "owner", "during", "century", "era", "period", "decade", "decades", "later", "years",
    "generation", "children", "people", "society", "political", "politics", "civil", "rights", "church", "school",
    "university", "college", "award", "awarded", "published", "publication", "writer", "author", "employed", "job",
    "trade", "industry", "market", "campaign", "election", "soldier", "army", "military", "regiment", "battle",
    "emigrated", "immigrant", "returned", "traveled", "travelled", "visited", "wrote", "recalled", "remembered",
    "mastered", "learned", "learning", "engagement", "would", "had", "was", "were", "did", "became", "began",
}
STRONG_PRONOUNS = {"he", "she", "they", "who", "him", "whom"}
WEAK_PRONOUNS = {"his", "her", "their", "them", "whose"}
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "st", "jr", "sr", "co", "inc", "ltd", "no", "nos", "vol", "ca", "c", "fig", "figs",
    "gen", "col", "capt", "lt", "rev", "ft", "mt", "ave", "u.s", "d.c", "e.g", "i.e", "vs", "p", "pp", "approx",
}

WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:st|nd|rd|th)?s?")
YEAR = re.compile(r"^(1[0-9]{3}|20[0-2][0-9])s?$")
ORDINAL_CENTURY = re.compile(r"\b\d+(st|nd|rd|th)[- ]century\b", re.IGNORECASE)
BOUNDARY = re.compile(r"([.!?][\"”’)]?)\s+(?=[\"“(]?[A-Z0-9])")


@dataclasses.dataclass(frozen=True)
class SentenceScore:
    text: str
    visual: float
    history: float
    tokens: int

    @property
    def score(self) -> float:
        return (self.visual - self.history) / max(self.tokens, 1)


class VisualTextExtractor:
    def __init__(self, config: VisualDescriptionExtractingMapperConfig) -> None:
        self.config = config
        self.visual_terms = COLORS | MATERIALS | DECORATION | SHAPES | COMPOSITION | APPEARANCE | OBJECTS | {
            t.lower() for t in config.extra_visual_terms
        }
        self.history_terms = HISTORY | {t.lower() for t in config.extra_history_terms}

    @staticmethod
    def sentences(paragraph: str) -> list[str]:
        pieces = []
        start = 0
        for match in BOUNDARY.finditer(paragraph):
            last_word = paragraph[start:match.start(1)].split(" ")[-1].lower()
            if last_word in ABBREVIATIONS or (len(last_word) == 1 and last_word.isalpha()):
                continue

            pieces.append(paragraph[start:match.end(1)].strip())
            start = match.end()

        pieces.append(paragraph[start:].strip())
        return [p for p in pieces if p]

    def score(self, sentence: str) -> SentenceScore:
        words = WORD.findall(sentence)
        visual = history = 0.0
        proper_nouns = 0
        for i, word in enumerate(words):
            lower = word.lower()
            if lower in self.visual_terms:
                visual += 1.0
            elif lower in PRESENT:
                visual += 0.5
            elif YEAR.match(lower):
                history += 1.5
            elif lower in self.history_terms or lower in STRONG_PRONOUNS:
                history += 1.0
            elif lower in WEAK_PRONOUNS or lower.endswith("ed") and len(lower) > 4:
                history += 0.5
            elif i > 0 and word[0].isupper() and proper_nouns < 3:
                proper_nouns += 1
                history += 0.5

        history += len(ORDINAL_CENTURY.findall(sentence))
        return SentenceScore(sentence, visual, history, len(words))

    def extract(self, text: str | None) -> tuple[str | None, float | None]:
        if not text:
            return None, None

        kept_paragraphs = []
        for paragraph in text.split("\n\n"):
            scores = [self.score(s) for s in self.sentences(paragraph)]
            if not scores:
                continue

            paragraph_score = sum(s.visual - s.history for s in scores) / max(sum(s.tokens for s in scores), 1)
            if paragraph_score > self.config.paragraph_threshold:
                kept = [s for s in scores if s.score > self.config.sentence_drop_threshold]
            else:
                kept = [s for s in scores if s.score >= self.config.sentence_keep_threshold]

            if kept:
                kept_paragraphs.append(" ".join(s.text for s in kept))

        visual = "\n\n".join(kept_paragraphs)
        return visual or None, min(len(visual) / len(text), 1.0)
