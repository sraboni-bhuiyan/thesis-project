from pathlib import Path
from pypdf import PdfReader
import json
import re

RAW_DIR = Path("data/guidelines_raw")
OUT_FILE = Path("data/guidelines_clean/corpus.jsonl")

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def clean_text(text):
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_into_chunks(text, min_words=200, max_words=400):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < min_words and chunks:
            chunks[-1] += " " + " ".join(chunk_words)
        else:
            chunks.append(" ".join(chunk_words))
        start = end
    return chunks

def build_corpus():
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDF files in {RAW_DIR}")

    with open(OUT_FILE, "w", encoding="utf-8") as out:
        total_chunks = 0
        for pdf_path in pdfs:
            print(f"Processing {pdf_path.name}")
            raw_text = extract_pdf_text(pdf_path)
            print(f"Extracted {len(raw_text.split())} words")
            cleaned = clean_text(raw_text)
            chunks = split_into_chunks(cleaned)
            print(f"Created {len(chunks)} chunks")

            for i, chunk in enumerate(chunks, start=1):
                record = {
                    "doc_id": pdf_path.stem.lower(),
                    "chunk_id": f"{pdf_path.stem.lower()}_{i:02d}",
                    "text": chunk,
                    "source": pdf_path.stem,
                    "section": "unknown",
                    "urgency_terms": [],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Total chunks written: {total_chunks}")

if __name__ == "__main__":
    build_corpus()