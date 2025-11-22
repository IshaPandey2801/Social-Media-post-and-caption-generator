# clean_data.py
"""
Cleans output/result.csv and creates JSONL train/valid files for fine-tuning.

Expect original CSV at: output/result.csv
Outputs:
 - output/cleaned_results.csv
 - output/train.jsonl
 - output/valid.jsonl
"""

import os
import csv
import time
import argparse
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

# Simple profanity list (extend as needed)
PROFANITY = {
    "badword1", "badword2"  # replace with real words you want to filter
}

STOP_DUPLICATE_SIMILARITY = 0.92  # threshold for fuzzy duplicates

ROOT = Path.cwd()
IN_CSV = ROOT / "output" / "result.csv"
CLEANED_CSV = ROOT / "output" / "cleaned_results.csv"
TRAIN_JSONL = ROOT / "output" / "train.jsonl"
VALID_JSONL = ROOT / "output" / "valid.jsonl"

def is_profane(text):
    if not isinstance(text, str): return False
    txt = text.lower()
    for p in PROFANITY:
        if p in txt:
            return True
    return False

def fuzzy_duplicate(a, b):
    return SequenceMatcher(None, a, b).ratio() >= STOP_DUPLICATE_SIMILARITY

def deduplicate_rows(rows):
    kept = []
    captions = []
    for r in rows:
        final = (r.get("final_caption") or "").strip()
        if not final:
            continue
        # skip profanity
        if is_profane(final):
            continue
        duplicate = False
        for cap in captions:
            if fuzzy_duplicate(final, cap):
                duplicate = True
                break
        if not duplicate:
            kept.append(r)
            captions.append(final)
    return kept

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    df = pd.read_csv(path)
    # standardize column names if necessary
    df.columns = [c.strip() for c in df.columns]
    return df

def make_example(row):
    base = (row.get("base_caption") or "").strip()
    final = (row.get("final_caption") or "").strip()
    tone = (row.get("tone") or "").strip()
    # Use platform as additional context
    platform = (row.get("platform") or "").strip()
    prompt_text = (row.get("prompt_text") or "").strip()

    # Input prompt for model: include base caption, tone, platform
    input_text = f"Rewrite the caption: \"{base}\" Tone: {tone} Platform: {platform}"
    # Optionally include prompt_text if base is empty
    if not base and prompt_text:
        input_text = f"Rewrite the prompt: \"{prompt_text}\" Tone: {tone} Platform: {platform}"

    return {"input": input_text, "output": final}

def split_train_valid(examples, valid_frac=0.1, seed=42):
    # simple deterministic split
    N = len(examples)
    if N == 0:
        return [], []
    # shuffle deterministically
    import random
    random.seed(seed)
    idxs = list(range(N))
    random.shuffle(idxs)
    n_valid = max(1, int(N * valid_frac))
    valid_idx = set(idxs[:n_valid])
    train = [examples[i] for i in range(N) if i not in valid_idx]
    valid = [examples[i] for i in range(N) if i in valid_idx]
    return train, valid

def run(valid_frac=0.1):
    print("Loading CSV:", IN_CSV)
    df = load_csv(IN_CSV)
    # ensure required columns exist
    expected = {"base_caption", "final_caption", "tone", "platform", "prompt_text"}
    missing = expected - set(df.columns)
    if missing:
        print("Warning: CSV missing columns:", missing)
    rows = df.to_dict(orient="records")
    print(f"Loaded {len(rows)} rows from CSV")

    # Filter rows with basic cleaning
    cleaned = []
    for r in rows:
        # remove rows where final_caption empty
        final = (r.get("final_caption") or "").strip()
        if not final:
            continue
        # remove very short captions
        if len(final) < 5:
            continue
        cleaned.append(r)
    print(f"After removing empty/short final captions: {len(cleaned)} rows")

    # Deduplicate fuzzy
    deduped = deduplicate_rows(cleaned)
    print(f"After fuzzy deduplication & profanity filter: {len(deduped)} rows")

    # Build examples
    examples = [make_example(r) for r in deduped]

    # split train/valid
    train_ex, valid_ex = split_train_valid(examples, valid_frac=valid_frac)
    print(f"Train examples: {len(train_ex)}, Valid examples: {len(valid_ex)}")

    # Save cleaned CSV
    pd.DataFrame(deduped).to_csv(CLEANED_CSV, index=False)
    print("Saved cleaned CSV to", CLEANED_CSV)

    # Save JSONL
    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for d in data:
                import json
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    write_jsonl(TRAIN_JSONL, train_ex)
    write_jsonl(VALID_JSONL, valid_ex)
    print("Wrote train.jsonl and valid.jsonl to", TRAIN_JSONL.parent)
    return TRAIN_JSONL, VALID_JSONL

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid_frac", type=float, default=0.1, help="Fraction of examples used for validation")
    args = parser.parse_args()
    run(valid_frac=args.valid_frac)
