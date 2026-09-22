"""
Demographic text transforms on MIMIC HPI text.

swap_demographics() rewrites gender/age cues for counterfactual variants;
mask_demographics() neutralizes them for C3.

MIMIC de-identifies ages as '___', so the cues that matter are titles, pronouns, sex words
and 'M'/'F' shorthand.
"""
import re

# (male form, female form) - whole word, case preserved
_GENDER_PAIRS = [
    ("mr", "ms"), ("mr", "mrs"), ("mr", "miss"),
    ("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
    ("man", "woman"), ("men", "women"), ("male", "female"), ("males", "females"),
    ("gentleman", "lady"), ("boy", "girl"),
]
# Relationship words (husband, mother, ...) describe other people: left unchanged on purpose.
# 'her' is ambiguous; -> 'his' before a noun, else 'him'.
_TO_FEMALE = {m: f for m, f in _GENDER_PAIRS if (m, f) not in {("mr", "mrs"), ("mr", "miss")}}
_TO_MALE = {f: m for m, f in _GENDER_PAIRS if f != "her"}

# Neutral forms for masking
_NEUTRAL = {
    "mr": "Pt", "ms": "Pt", "mrs": "Pt", "miss": "Pt",
    "he": "they", "she": "they", "him": "them", "his": "their", "her": "their",
    "himself": "themself", "herself": "themself",
    "man": "person", "woman": "person", "men": "people", "women": "people",
    "male": "patient", "female": "patient", "males": "patients", "females": "patients",
    "gentleman": "person", "lady": "person", "boy": "child", "girl": "child",
}

_WORD = re.compile(r"\b([A-Za-z]+)\b(\.?)")
# Matches "___ M", "___ y/o F", "65 yo M". An age token is required: a bare number before M/F
# is NOT matched, or vitals like "101 F" would be read as a female patient.
_SEX_LETTER = re.compile(r"((?:___\s*(?:y/?o|yo|yr old|year[- ]old)?|\b\d{1,3}\s*(?:y/?o|yo|yr old|year[- ]old))\s*)\b([MF])\b")
_AGE_PHRASE = re.compile(r"\b\d{1,3}\s*(?:-| )?\s*(?:y/?o|yo|yrs?(?:\s*old)?|years?(?:[- ]old)?)\b", re.IGNORECASE)
_AGE_SEX_COMPACT = re.compile(r"\b(\d{1,3})([MF])\b")


def _match_case(src: str, repl: str) -> str:
    if src.isupper() and len(src) > 1:
        return repl.upper()
    if src[0].isupper():
        return repl[0].upper() + repl[1:]
    return repl


def _her_is_possessive(text: str, m) -> bool:
    """'her pain' -> possessive ('his'); 'wake her.' / 'gave her to' -> object ('him')."""
    if m.group(2) or text[m.end():m.end() + 1] not in (" ", ""):
        return False
    nxt = text[m.end():m.end() + 20].lstrip()
    return nxt[:1].isalpha() and not re.match(
        r"(to|and|or|in|at|on|with|for|from|that|as|is|was|by|up|down|out|home|back)\b", nxt, re.I)


def _swap_gender(text: str, target: str) -> str:
    table = _TO_MALE if target == "male" else _TO_FEMALE

    def repl(m):
        word, dot = m.group(1), m.group(2)
        lw = word.lower()
        if target == "male" and lw == "her":
            return _match_case(word, "his" if _her_is_possessive(text, m) else "him") + dot
        if lw in table:
            return _match_case(word, table[lw]) + dot
        return m.group(0)

    text = _WORD.sub(repl, text)
    letter = "M" if target == "male" else "F"
    text = _SEX_LETTER.sub(lambda m: m.group(1) + letter, text)
    text = _AGE_SEX_COMPACT.sub(lambda m: m.group(1) + letter, text)
    return text


def swap_demographics(text: str, target_gender: str, target_age: int, orig_age=None) -> str:
    """Rewrite gender cues to `target_gender` ('male'/'female') and explicit ages to `target_age`."""
    if not text:
        return text
    text = _swap_gender(text, target_gender.lower())
    text = _AGE_PHRASE.sub(lambda m: re.sub(r"\d{1,3}", str(target_age), m.group(0), count=1), text)
    text = _AGE_SEX_COMPACT.sub(lambda m: f"{target_age}{m.group(2)}", text)
    if orig_age is not None:
        text = re.sub(rf"\b{orig_age}\b(?=\s*(?:-|\s)?(?:y|year))", str(target_age), text)
    return text


def mask_demographics(text: str) -> str:
    """Remove gender and age cues from free text (C3 masking)."""
    if not text:
        return text

    def repl(m):
        word, dot = m.group(1), m.group(2)
        lw = word.lower()
        if lw in _NEUTRAL:
            new = _NEUTRAL[lw]
            if lw == "her" and not _her_is_possessive(text, m):
                new = "them"
            if lw in {"he", "she"}:
                new = "the patient"
            # titles like "Ms. ___" -> "Pt ___"
            return _match_case(word, new) + ("" if lw in {"mr", "ms", "mrs", "miss"} else dot)
        return m.group(0)

    text = _AGE_SEX_COMPACT.sub("___", text)
    text = _SEX_LETTER.sub(lambda m: m.group(1).rstrip() + " ", text)
    text = _AGE_PHRASE.sub("___ y/o", text)
    text = _WORD.sub(repl, text)
    return text
