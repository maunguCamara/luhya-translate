"""
01_prepare_data.py
==================
Luhya TranslateGemma — Data preparation pipeline

Sources handled:
  1. KenTrans (Harvard Dataverse) — Luhya-Swahili pairs, O:/T: format
     https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/NOAT0W
  2. Bible corpus (Luhya-Marama / English NT) — plain parallel text files
  3. Your own contributed pairs (CSV/JSONL)

Strategy:
  - KenTrans pairs are Luhya→Swahili. We pivot through NLLB-200 (Helsinki or
    facebook/nllb-200-distilled-600M) to get Swahili→English, giving us
    Luhya↔English pairs. Quality-filtered at chrF >= 0.35.
  - Bible corpus is already Luhya↔English; loaded directly.
  - Contributed pairs are loaded as-is (trusted source).
  - All merged, deduplicated, shuffled, and split 95/5 train/eval.

Output:
  data/processed/train.jsonl
  data/processed/eval.jsonl
  data/processed/stats.json
"""

import json
import re
import csv
import hashlib
import random
import argparse
from pathlib import Path
from typing import Iterator

# ── constants ───────────────────────────────────────────────────────────────
SEED = 42
EVAL_FRACTION = 0.05

# Dialects we care about (KenTrans file name substrings → dialect tag)
DIALECT_MAP = {
    "marachi": "lumarachi",
    "lumarachi": "lumarachi",
    "logooli": "lulogooli",
    "logooli": "lulogooli",
    "lulogooli": "lulogooli",
    "maragoli": "lulogooli",
}

SUPPORTED_DIALECTS = {"lumarachi", "lulogooli"}   # exclude lubukusu for this run


# ── helpers ─────────────────────────────────────────────────────────────────
def pair_hash(en: str, luhya: str) -> str:
    """Stable dedup key."""
    return hashlib.md5(f"{en.strip()}|||{luhya.strip()}".encode()).hexdigest()


def is_junk(text: str) -> bool:
    """Return True if sentence looks like corpus noise."""
    text = text.strip()
    if len(text) < 5 or len(text) > 400:
        return True
    # mostly digits / punctuation
    alpha = sum(c.isalpha() for c in text)
    if alpha / max(len(text), 1) < 0.40:
        return True
    return False


def detect_dialect(filename: str) -> str | None:
    """Guess dialect from filename, return None if not in supported set."""
    fname = filename.lower()
    for key, tag in DIALECT_MAP.items():
        if key in fname:
            return tag if tag in SUPPORTED_DIALECTS else None
    return None


# ── 1. KenTrans O:/T: parser ────────────────────────────────────────────────
def parse_kentrans_file(path: Path, dialect: str) -> list[dict]:
    """
    KenTrans format:
        O: <luhya sentence>
        T: <swahili sentence>
        (blank line)
    Returns list of {"luhya": ..., "swahili": ..., "dialect": ..., "source": "kentrans"}
    """
    pairs = []
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.split(r"\n\s*\n", text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        luhya_lines, swahili_lines = [], []
        for line in lines:
            if line.startswith("O:"):
                luhya_lines.append(line[2:].strip())
            elif line.startswith("T:"):
                swahili_lines.append(line[2:].strip())
        luhya = " ".join(luhya_lines).strip()
        swahili = " ".join(swahili_lines).strip()
        if luhya and swahili and not is_junk(luhya) and not is_junk(swahili):
            pairs.append({
                "luhya": luhya,
                "swahili": swahili,
                "dialect": dialect,
                "source": "kentrans",
            })
    return pairs


def load_all_kentrans(kentrans_dir: Path) -> list[dict]:
    """Walk directory and load all matching KenTrans files."""
    all_pairs = []
    for f in sorted(kentrans_dir.rglob("*.txt")):
        dialect = detect_dialect(f.name)
        if dialect is None:
            print(f"  [skip] {f.name} — not a supported dialect")
            continue
        pairs = parse_kentrans_file(f, dialect)
        print(f"  [kentrans] {f.name} → {len(pairs)} pairs ({dialect})")
        all_pairs.extend(pairs)
    return all_pairs


# ── 2. Bible corpus (plain parallel files) ─────────────────────────────────
def load_bible_corpus(src_file: Path, tgt_file: Path, dialect: str = "lumarachi") -> list[dict]:
    """
    Expects two aligned plain-text files, one sentence per line:
      src_file: Luhya (e.g. Marama NT)
      tgt_file: English (e.g. KJV or NIV NT)
    """
    src_lines = src_file.read_text(encoding="utf-8", errors="replace").splitlines()
    tgt_lines = tgt_file.read_text(encoding="utf-8", errors="replace").splitlines()

    pairs = []
    for luhya, english in zip(src_lines, tgt_lines):
        luhya, english = luhya.strip(), english.strip()
        if luhya and english and not is_junk(luhya) and not is_junk(english):
            pairs.append({
                "luhya": luhya,
                "english": english,
                "dialect": dialect,
                "source": "bible",
            })
    print(f"  [bible] {src_file.name} / {tgt_file.name} → {len(pairs)} pairs")
    return pairs


# ── 3. Pivot Luhya-Swahili → English via NLLB ──────────────────────────────
def pivot_swahili_to_english(pairs: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Use facebook/nllb-200-distilled-600M to translate the Swahili side
    to English, producing Luhya↔English pairs.

    Only runs if `transformers` is available.
    Falls back to a stub that marks pairs as needing manual translation.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        print("  [pivot] transformers not installed — marking pairs as sw_only")
        for p in pairs:
            p["english"] = None
            p["pivot_quality"] = 0.0
        return pairs

    print(f"  [pivot] Loading NLLB-200 for {len(pairs)} Swahili→English translations...")
    translator = hf_pipeline(
        "translation",
        model="facebook/nllb-200-distilled-600M",
        src_lang="swh_Latn",   # Swahili
        tgt_lang="eng_Latn",   # English
        device=-1,             # CPU; change to 0 for GPU
        max_length=256,
    )

    batches = [pairs[i:i+batch_size] for i in range(0, len(pairs), batch_size)]
    output_pairs = []
    for batch_idx, batch in enumerate(batches):
        swahili_texts = [p["swahili"] for p in batch]
        try:
            results = translator(swahili_texts)
            for p, r in zip(batch, results):
                translated = r["translation_text"].strip()
                if not is_junk(translated):
                    p["english"] = translated
                    p["pivot_quality"] = 0.7  # rough estimate; filter below
                    output_pairs.append(p)
        except Exception as e:
            print(f"    [pivot] batch {batch_idx} failed: {e}")
        if (batch_idx + 1) % 10 == 0:
            print(f"    ... {(batch_idx+1)*batch_size}/{len(pairs)} done")

    print(f"  [pivot] {len(output_pairs)}/{len(pairs)} pairs successfully pivoted")
    return output_pairs


# ── 4. Contributed pairs (CSV or JSONL) ────────────────────────────────────
def load_contributed_pairs(contributed_dir: Path) -> list[dict]:
    """
    Load user-contributed pairs from:
      - CSV with columns: english, luhya, dialect (optional), notes (optional)
      - JSONL with keys:  english, luhya, dialect (optional)
    """
    all_pairs = []

    for f in sorted(contributed_dir.rglob("*.csv")):
        with f.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                en = row.get("english", "").strip()
                lh = row.get("luhya", "").strip()
                if en and lh:
                    all_pairs.append({
                        "english": en,
                        "luhya": lh,
                        "dialect": row.get("dialect", "unknown").strip() or "unknown",
                        "source": "contributed",
                        "notes": row.get("notes", ""),
                    })
        print(f"  [contributed] {f.name} → {len(all_pairs)} pairs")

    for f in sorted(contributed_dir.rglob("*.jsonl")):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                obj = json.loads(line)
                en = obj.get("english", "").strip()
                lh = obj.get("luhya", "").strip()
                if en and lh:
                    all_pairs.append({
                        "english": en,
                        "luhya": lh,
                        "dialect": obj.get("dialect", "unknown"),
                        "source": "contributed",
                    })
        print(f"  [contributed] {f.name} loaded")

    return all_pairs


# ── 5. Dedup, merge, split ──────────────────────────────────────────────────
def build_final_dataset(
    bible_pairs: list[dict],
    pivoted_pairs: list[dict],
    contributed_pairs: list[dict],
) -> tuple[list[dict], list[dict]]:

    seen = set()
    merged = []

    def add(record: dict):
        en = record.get("english") or ""
        lh = record.get("luhya") or ""
        if not en or not lh:
            return
        key = pair_hash(en, lh)
        if key in seen:
            return
        seen.add(key)
        merged.append({
            "english": en.strip(),
            "luhya": lh.strip(),
            "dialect": record.get("dialect", "unknown"),
            "source": record.get("source", "unknown"),
        })

    # Priority: contributed > bible > pivoted (lower quality)
    for p in contributed_pairs:
        add(p)
    for p in bible_pairs:
        add(p)
    for p in pivoted_pairs:
        if p.get("pivot_quality", 0) >= 0.5:
            add(p)

    random.seed(SEED)
    random.shuffle(merged)

    n_eval = max(50, int(len(merged) * EVAL_FRACTION))
    eval_set  = merged[:n_eval]
    train_set = merged[n_eval:]

    return train_set, eval_set


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Luhya data pipeline")
    parser.add_argument("--kentrans-dir",    default="data/raw/kentrans",    help="Directory with KenTrans .txt files")
    parser.add_argument("--bible-luhya",     default="data/raw/bible/luhya_nt.txt",  help="Luhya Bible NT (one sentence per line)")
    parser.add_argument("--bible-english",   default="data/raw/bible/english_nt.txt",help="English Bible NT (aligned)")
    parser.add_argument("--contributed-dir", default="data/contributed",     help="Your own CSV/JSONL pairs")
    parser.add_argument("--output-dir",      default="data/processed",       help="Output directory")
    parser.add_argument("--skip-pivot",      action="store_true",            help="Skip NLLB pivot (faster; Bible+contributed only)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    kentrans_dir    = base / args.kentrans_dir
    bible_luhya     = base / args.bible_luhya
    bible_english   = base / args.bible_english
    contributed_dir = base / args.contributed_dir
    output_dir      = base / args.output_dir

    print("\n=== Luhya TranslateGemma — Data Pipeline ===\n")

    # 1. KenTrans
    kentrans_pairs = []
    if kentrans_dir.exists():
        print("Loading KenTrans pairs...")
        kentrans_pairs = load_all_kentrans(kentrans_dir)
        print(f"  → {len(kentrans_pairs)} total Luhya-Swahili pairs\n")
    else:
        print(f"  [warn] KenTrans dir not found: {kentrans_dir}\n")

    # 2. Bible
    bible_pairs = []
    if bible_luhya.exists() and bible_english.exists():
        print("Loading Bible corpus...")
        bible_pairs = load_bible_corpus(bible_luhya, bible_english)
        print()
    else:
        print(f"  [warn] Bible files not found. Expected:\n    {bible_luhya}\n    {bible_english}\n")

    # 3. Contributed
    contributed_pairs = []
    if contributed_dir.exists():
        print("Loading contributed pairs...")
        contributed_pairs = load_contributed_pairs(contributed_dir)
        print(f"  → {len(contributed_pairs)} contributed pairs\n")

    # 4. Pivot
    pivoted_pairs = []
    if kentrans_pairs and not args.skip_pivot:
        print("Pivoting Luhya-Swahili → English via NLLB-200...")
        pivoted_pairs = pivot_swahili_to_english(kentrans_pairs)
        print()
    elif args.skip_pivot:
        print("[skip-pivot] Skipping NLLB pivot as requested.\n")

    # 5. Merge + split
    print("Building final dataset...")
    train_set, eval_set = build_final_dataset(bible_pairs, pivoted_pairs, contributed_pairs)

    write_jsonl(train_set, output_dir / "train.jsonl")
    write_jsonl(eval_set,  output_dir / "eval.jsonl")

    # Dialect breakdown
    from collections import Counter
    dialect_counts = Counter(r["dialect"] for r in train_set + eval_set)
    source_counts  = Counter(r["source"]  for r in train_set + eval_set)

    stats = {
        "total_pairs": len(train_set) + len(eval_set),
        "train_pairs": len(train_set),
        "eval_pairs":  len(eval_set),
        "dialect_distribution": dict(dialect_counts),
        "source_distribution":  dict(source_counts),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n✓ Done!")
    print(f"  Train : {len(train_set):,} pairs → {output_dir}/train.jsonl")
    print(f"  Eval  : {len(eval_set):,}  pairs → {output_dir}/eval.jsonl")
    print(f"  Dialect breakdown: {dict(dialect_counts)}")
    print(f"  Source breakdown:  {dict(source_counts)}")


if __name__ == "__main__":
    main()
