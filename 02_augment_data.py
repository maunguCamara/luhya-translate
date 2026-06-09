"""
02_augment_data.py
==================
Data augmentation for Luhya TranslateGemma.

For a low-resource language with ~8–11k pairs, augmentation can meaningfully
boost coverage. Techniques applied:

  1. Back-translation noise  — translate English→intermediate lang→English,
     creating paraphrase variants. Keeps original Luhya, new English source.
     This teaches the model robustness to English phrasing variation.

  2. Dialect mixing (soft)   — for pairs that exist in BOTH dialects for the
     same concept, we add cross-dialect annotations. Helps model learn shared
     Luhya structure across Marachi/Maragoli.

  3. Length bucketing         — report sentence-length distribution; ensures
     train split has coverage across short (< 8 tok), medium, and long (> 25 tok)
     sentences. Warns if a bucket is under-represented.

  4. Sentence sampling        — for very large pivoted sets, sample top-quality
     pairs by length ratio heuristic (en_len / lh_len should be 0.5–2.0).

Output:
  data/augmented/train_augmented.jsonl
  data/augmented/augmentation_report.json
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

# Length ratio quality filter
MIN_RATIO = 0.4   # luhya can't be much shorter than english
MAX_RATIO = 2.8   # or much longer


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── 1. Length ratio filter ──────────────────────────────────────────────────
def length_ratio_filter(records: list[dict]) -> tuple[list[dict], int]:
    kept, dropped = [], 0
    for r in records:
        en_tok = len(r["english"].split())
        lh_tok = len(r["luhya"].split())
        if lh_tok == 0:
            dropped += 1
            continue
        ratio = en_tok / lh_tok
        if MIN_RATIO <= ratio <= MAX_RATIO:
            kept.append(r)
        else:
            dropped += 1
    return kept, dropped


# ── 2. Length bucket analysis ───────────────────────────────────────────────
def bucket_analysis(records: list[dict]) -> dict:
    buckets = defaultdict(int)
    for r in records:
        n = len(r["english"].split())
        if n < 8:
            buckets["short (< 8)"] += 1
        elif n < 20:
            buckets["medium (8–19)"] += 1
        elif n < 35:
            buckets["long (20–34)"] += 1
        else:
            buckets["very_long (35+)"] += 1
    return dict(buckets)


# ── 3. Back-translation via NLLB ────────────────────────────────────────────
def back_translate(records: list[dict], n_samples: int = 500, batch_size: int = 32) -> list[dict]:
    """
    For n_samples pairs: English → French (via NLLB) → English
    Creates a paraphrase variant. Keeps the original Luhya.
    Adds augmented=True flag so we can separate them later.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("  [aug] transformers not installed, skipping back-translation")
        return []

    sample = random.sample(records, min(n_samples, len(records)))

    print(f"  [aug] Back-translating {len(sample)} pairs en→fr→en ...")
    en_to_fr = hf_pipeline(
        "translation", model="Helsinki-NLP/opus-mt-en-fr",
        device=-1, max_length=200,
    )
    fr_to_en = hf_pipeline(
        "translation", model="Helsinki-NLP/opus-mt-fr-en",
        device=-1, max_length=200,
    )

    augmented = []
    for i in range(0, len(sample), batch_size):
        batch = sample[i:i+batch_size]
        try:
            fr_results = en_to_fr([r["english"] for r in batch])
            fr_texts   = [r["translation_text"] for r in fr_results]
            en2_results = fr_to_en(fr_texts)
            for orig, en2 in zip(batch, en2_results):
                new_en = en2["translation_text"].strip()
                if new_en and new_en.lower() != orig["english"].lower():
                    augmented.append({
                        "english": new_en,
                        "luhya":   orig["luhya"],
                        "dialect": orig["dialect"],
                        "source":  "back_translation",
                        "augmented": True,
                    })
        except Exception as e:
            print(f"    [aug] batch {i//batch_size} failed: {e}")

    print(f"  [aug] Generated {len(augmented)} back-translated pairs")
    return augmented


# ── 4. Dialect pair cross-annotation ────────────────────────────────────────
def find_cross_dialect_pairs(records: list[dict]) -> dict:
    """
    Group by English source to find sentences that appear in both dialects.
    Returns stats (we don't auto-generate cross-pairs — too risky without native review).
    """
    from collections import defaultdict
    en_to_dialects = defaultdict(set)
    for r in records:
        en_to_dialects[r["english"].lower().strip()].add(r["dialect"])
    multi = {k: v for k, v in en_to_dialects.items() if len(v) > 1}
    return {
        "sentences_in_multiple_dialects": len(multi),
        "dialects_seen": list({d for v in en_to_dialects.values() for d in v}),
    }


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        default="data/processed/train.jsonl")
    parser.add_argument("--output-dir",   default="data/augmented")
    parser.add_argument("--back-translate", action="store_true", help="Run back-translation (slow)")
    parser.add_argument("--bt-samples",   type=int, default=500)
    args = parser.parse_args()

    base  = Path(__file__).parent.parent
    inp   = base / args.input
    outdir = base / args.output_dir

    print("\n=== Luhya TranslateGemma — Augmentation ===\n")
    records = load_jsonl(inp)
    print(f"Loaded {len(records):,} training pairs")

    # Length ratio filter
    records, n_dropped = length_ratio_filter(records)
    print(f"After length-ratio filter: {len(records):,} kept, {n_dropped} dropped")

    # Bucket analysis
    buckets = bucket_analysis(records)
    print(f"Length distribution: {buckets}")
    for bucket, count in buckets.items():
        frac = count / len(records)
        if frac < 0.05:
            print(f"  ⚠ Bucket '{bucket}' is under-represented ({count} = {frac:.1%}). Consider adding more pairs.")

    # Cross-dialect stats
    xdial = find_cross_dialect_pairs(records)
    print(f"Cross-dialect overlap: {xdial}")

    # Back-translation (optional, slow)
    aug_pairs = []
    if args.back_translate:
        aug_pairs = back_translate(records, n_samples=args.bt_samples)

    all_records = records + aug_pairs
    random.shuffle(all_records)

    write_jsonl(all_records, outdir / "train_augmented.jsonl")

    report = {
        "original_pairs": len(records),
        "augmented_pairs": len(aug_pairs),
        "total": len(all_records),
        "length_buckets": buckets,
        "cross_dialect": xdial,
        "length_ratio_dropped": n_dropped,
    }
    (outdir / "augmentation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print(f"\n✓ Augmentation done!")
    print(f"  Total pairs for training: {len(all_records):,}")
    print(f"  Output: {outdir}/train_augmented.jsonl")


if __name__ == "__main__":
    main()
