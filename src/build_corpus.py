"""
Build the retrieval corpus: data/guidelines_raw/*.pdf -> data/guidelines_clean/corpus.jsonl

Extracts PDF text, strips headers/footers, navigation boilerplate and training-exercise
vignettes, then chunks to 250 words with 50-word overlap.

Source PDFs are copyright-restricted (CAEP/CTAS NWG; ENA): local research use only, never
redistributed. data/ is gitignored for this reason.

Usage:
  python src/build_corpus.py            # build
  python src/build_corpus.py --verify   # rebuild to temp and compare sha256, no overwrite
"""
import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import pypdf
from pypdf import PdfReader

from config import CORPUS_FILE, GUIDELINES_PATH

CHUNK_WORDS = 250
OVERLAP_WORDS = 50
MIN_WORDS = 100          # a trailing chunk shorter than this is merged into the previous one
HEADER_FOOTER_RATIO = 0.5  # a line on more than this share of pages is boilerplate

# Text that survives extraction but carries no clinical content.
BOILERPLATE = [
    r"skip navigation", r"sidebar menu", r"sidebar contact person", r"sidebar location",
    r"focus area and clinic finder", r"construction cku \d+",
]

# Mojibake from these PDFs: ENA(R), PARTICIPANT(')S etc. extract as U+FFFD.
_ARTIFACTS = {"�": "", "’": "'", "‘": "'", "“": '"', "”": '"',
              "–": "-", "—": "-", " ": " "}

URGENCY_VOCAB = [
    "ctas 1", "ctas 2", "ctas 3", "ctas 4", "ctas 5", "level 1", "level 2", "level 3",
    "level 4", "level 5", "resuscitation", "emergent", "urgent", "less urgent", "non-urgent",
    "red", "orange", "yellow", "green", "blue", "immediate", "life-threatening", "acuity",
]


def extract_pages(path: Path):
    """Text of each page. Empty pages kept as '' so page ratios stay honest."""
    return [(page.extract_text() or "") for page in PdfReader(str(path)).pages]


def drop_repeated_lines(pages):
    """Drop running heads and footers: lines on more than HEADER_FOOTER_RATIO of pages."""
    if len(pages) < 3:                      # too few pages for the ratio to mean anything
        return pages
    counts = Counter()
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            counts[line] += 1
    cutoff = len(pages) * HEADER_FOOTER_RATIO
    repeated = {line for line, n in counts.items() if n > cutoff}
    return ["\n".join(ln for ln in page.splitlines() if ln.strip() not in repeated) for page in pages]


# Training exercises are unanswered patient vignettes; left in, they feed the model unlabelled
# cases as guidance. Isolated numbered sentences (GCS criteria) are real content, so only runs of
# MIN_RUN consecutively numbered items are excised.
_VIGNETTE_ITEM = re.compile(r"\b(\d{1,2})\.\s+(?:A|An|The|Patient|Pt)\b")
_SECTION_HEAD = re.compile(r"\s\d{0,3}\s*\d+\.\d+\s+[A-Z]")
_EXERCISE_CUE = re.compile(r"\b(identify|following patient|complete the|match the|for the following)\b", re.I)
MIN_RUN = 4
MAX_ITEM_GAP = 350


def strip_exercise_lists(text: str) -> str:
    """Remove unanswered training-exercise vignettes."""
    items = [(m.start(), int(m.group(1))) for m in _VIGNETTE_ITEM.finditer(text)]
    runs, current = [], []
    for pos, num in items:
        if current and num == current[-1][1] + 1 and pos - current[-1][0] <= MAX_ITEM_GAP:
            current.append((pos, num))
        else:
            if len(current) >= MIN_RUN:
                runs.append(current)
            current = [(pos, num)]
    if len(current) >= MIN_RUN:
        runs.append(current)

    for run in reversed(runs):                      # reversed: earlier offsets stay valid
        start, last = run[0][0], run[-1][0]
        lead = text.rfind(". ", max(0, start - 300), start)   # take the instruction sentence too
        if lead != -1 and _EXERCISE_CUE.search(text[lead:start]):
            start = lead + 2
        head = _SECTION_HEAD.search(text, last)     # stop at the next numbered section heading
        end = head.start() if head and head.start() - last < 600 else min(len(text), last + 400)
        text = text[:start] + " " + text[end:]
    return re.sub(r"\s+", " ", text).strip()


def clean_text(text: str) -> str:
    """De-hyphenate, drop boilerplate, normalise artifacts and whitespace."""
    for bad, good in _ARTIFACTS.items():
        text = text.replace(bad, good)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)   # de-hyphenate across line breaks
    text = re.sub(r"\s+", " ", text)
    for pattern in BOILERPLATE:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_words: int, overlap: int, min_words: int):
    """Fixed-size word chunks with sliding overlap."""
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_words - overlap)
    chunks, start = [], 0
    while start < len(words):
        piece = words[start:start + chunk_words]
        if len(piece) < min_words and chunks:
            break                            # short tail already covered by the previous overlap
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
        start += step
    return chunks


# Course scaffolding, dropped only when the chunk also lacks acuity and clinical vocabulary,
# so clinical text is never discarded.
_NON_CLINICAL = re.compile(
    r"\b(exercise|scenario \d|practice (case|question)|answer key|quiz|workshop|small group|"
    r"course (goals|organization|overview)|participant.s manual|learning objectives|module \d|"
    r"table of contents|acknowledg|acronyms)\b", re.I)
_CLINICAL = re.compile(
    r"\b(pain|bleed|breath|airway|chest|abdom|fever|trauma|vital sign|pulse|blood pressure|"
    r"conscious|sepsis|stroke|seizure|injury|wound|vomit|dyspnea|hypoten|tachycard|oxygen|"
    r"glasgow|triage level|undertriag|symptom)\b", re.I)
MIN_CLINICAL_HITS = 3


def is_non_clinical(text: str) -> bool:
    """Course scaffolding with no acuity or clinical vocabulary."""
    return (bool(_NON_CLINICAL.search(text))
            and not urgency_terms_in(text)
            and len(_CLINICAL.findall(text)) < MIN_CLINICAL_HITS)


def urgency_terms_in(text: str):
    low = text.lower()
    return sorted({term for term in URGENCY_VOCAB if re.search(rf"\b{re.escape(term)}\b", low)})


def build_records(chunk_words: int, overlap: int, min_words: int, verbose: bool = True):
    pdfs = sorted(GUIDELINES_PATH.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {GUIDELINES_PATH}")
    records = []
    for path in pdfs:
        doc_id = path.stem.lower()
        pages = drop_repeated_lines(extract_pages(path))
        body = strip_exercise_lists(clean_text(" ".join(pages)))
        pieces = [p for p in split_into_chunks(body, chunk_words, overlap, min_words)
                  if not is_non_clinical(p)]
        for i, piece in enumerate(pieces, start=1):
            records.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{i:03d}",
                "text": piece,
                "source": path.stem,
                "section": "unknown",
                "urgency_terms": urgency_terms_in(piece),
            })
        if verbose:
            print(f"  {path.name:<40} {len(pages):>3} pages  {len(body.split()):>6} words  "
                  f"-> {len(pieces):>3} chunks")
    return records


def write_jsonl(records, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build the guideline retrieval corpus from PDFs.")
    parser.add_argument("--chunk-words", type=int, default=CHUNK_WORDS)
    parser.add_argument("--overlap", type=int, default=OVERLAP_WORDS)
    parser.add_argument("--min-words", type=int, default=MIN_WORDS)
    parser.add_argument("--output", default=None, help=f"Default: {CORPUS_FILE}")
    parser.add_argument("--verify", action="store_true",
                        help="Rebuild to a temp file and compare against the existing corpus; do not overwrite.")
    args = parser.parse_args()

    if args.overlap >= args.chunk_words:
        raise SystemExit("--overlap must be smaller than --chunk-words")

    out = Path(args.output) if args.output else CORPUS_FILE
    print(f"pypdf {pypdf.__version__} | chunk={args.chunk_words}w overlap={args.overlap}w "
          f"min={args.min_words}w")
    print(f"Reading PDFs from {GUIDELINES_PATH}")
    records = build_records(args.chunk_words, args.overlap, args.min_words)

    lengths = sorted(len(r["text"].split()) for r in records)
    with_urgency = sum(1 for r in records if r["urgency_terms"])
    print(f"\n{len(records)} chunks from {len({r['doc_id'] for r in records})} documents")
    print(f"  words per chunk: min {lengths[0]}, median {lengths[len(lengths) // 2]}, max {lengths[-1]}")
    print(f"  chunks containing urgency vocabulary: {with_urgency}/{len(records)} "
          f"({with_urgency / len(records):.0%})")

    if args.verify:
        if not out.exists():
            raise SystemExit(f"--verify: {out} does not exist yet; run without --verify first.")
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        write_jsonl(records, tmp_path)
        existing, rebuilt = sha256_of(out), sha256_of(tmp_path)
        tmp_path.unlink()
        print(f"\nexisting  {out.name}: sha256 {existing[:16]}")
        print(f"rebuilt            : sha256 {rebuilt[:16]}")
        print("MATCH - corpus is reproducible from the PDFs." if existing == rebuilt
              else "DIFFER - the committed corpus was not produced by these settings.")
        raise SystemExit(0 if existing == rebuilt else 1)

    write_jsonl(records, out)
    print(f"\nWrote {out}  (sha256 {sha256_of(out)[:16]})")
    print("Next: python src/build_index.py   (the FAISS index is now stale)")


if __name__ == "__main__":
    main()
