# paraphrase_data.py
"""
Paraphrase captions to augment dataset.

Usage:
    python paraphrase_data.py --n 3

Outputs:
 - output/train_augmented.jsonl  (same format as train.jsonl, contains original + paraphrases)
"""

import argparse
from pathlib import Path
import json
import pandas as pd
import os
import sys

# Choose model (local). If you have limited RAM/GPU use a small/efficient model.
PARAPHRASE_MODEL = "Vamsi/T5_Paraphrase_Paws"  # change if you want another
USE_API = False  # If True, script will try to call OpenAI instead (not implemented here)

def load_captions(csv_path):
    df = pd.read_csv(csv_path)
    # prefer cleaned_results if exists
    if "final_caption" not in df.columns:
        raise ValueError("CSV does not contain 'final_caption' column")
    captions = df["final_caption"].dropna().astype(str).tolist()
    return captions, df

def init_model_tokenizer(model_name):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model, tokenizer

def paraphrase_batch(model, tokenizer, texts, num_return_sequences=3, max_length=128, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    paraphrases_for_texts = []
    for t in texts:
        # build model-specific prompt
        input_text = "paraphrase: " + t + " </s>"
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=256).to(device)
        # generate
        outputs = model.generate(inputs, max_length=max_length, num_beams=10,
                                 num_return_sequences=num_return_sequences, temperature=1.5)
        decs = [tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=True) for o in outputs]
        # de-duplicate and strip
        seen = set()
        cleaned = []
        for d in decs:
            dd = d.strip()
            if dd.lower() not in seen and len(dd) > 3:
                cleaned.append(dd)
                seen.add(dd.lower())
        paraphrases_for_texts.append(cleaned)
    return paraphrases_for_texts

def main(n_paraphrases=3, input_csv=None, output_jsonl=None, sample_limit=None):
    root = Path.cwd()
    input_csv = Path(input_csv) if input_csv else root / "output" / "cleaned_results.csv"
    if not input_csv.exists():
        input_csv = root / "output" / "result.csv"
    if not input_csv.exists():
        print("No CSV found at output/cleaned_results.csv or output/result.csv. Run clean_data.py first.")
        sys.exit(1)

    captions, df = load_captions(input_csv)

    if sample_limit:
        captions = captions[:sample_limit]

    print(f"Loaded {len(captions)} captions from {input_csv}")

    # Load paraphrase model
    try:
        model, tokenizer = init_model_tokenizer(PARAPHRASE_MODEL)
    except Exception as e:
        print("Error loading paraphrase model:", e)
        print("Make sure transformers is installed and you have internet for first-time download.")
        raise

    # Generate paraphrases in batches
    batch_size = 8
    all_augmented = []  # list of dicts {"input":..., "output":...}
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        print(f"Paraphrasing batch {i}..{i+len(batch)-1}")
        paras = paraphrase_batch(model, tokenizer, batch, num_return_sequences=n_paraphrases)
        for base, p_list in zip(batch, paras):
            # add original as one example (if you want)
            all_augmented.append({"input": f"Rewrite the caption: \"{base}\" Tone: mixed Platform: any", "output": base})
            for p in p_list:
                all_augmented.append({"input": f"Rewrite the caption: \"{base}\" Tone: mixed Platform: any", "output": p})

    out_path = Path(output_jsonl) if output_jsonl else root / "output" / "train_augmented.jsonl"
    print("Writing augmented JSONL to", out_path)
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in all_augmented:
            w.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Done. Total augmented examples:", len(all_augmented))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3, help="paraphrases per caption")
    parser.add_argument("--sample", type=int, default=None, help="limit number of source captions (for quick test)")
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_jsonl", type=str, default=None)
    args = parser.parse_args()
    main(n_paraphrases=args.n, input_csv=args.input_csv, output_jsonl=args.output_jsonl, sample_limit=args.sample)
