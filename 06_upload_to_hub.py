"""
06_upload_to_hub.py
===================
Upload your English-Luhya dataset to HuggingFace Hub,
then generate a model card template.

Usage:
    huggingface-cli login
    python scripts/06_upload_to_hub.py \
        --dataset-id yourname/english-luhya-translations \
        --model-id   yourname/luhya_translategemma_4b_v1

This creates:
  - A dataset repo with your train/eval JSONL + README
  - A model card README.md template (fill in your BLEU/chrF++ scores)
"""

import json
import argparse
from pathlib import Path
from datetime import date


def upload_dataset(dataset_id: str, data_dir: Path):
    from datasets import Dataset, DatasetDict
    from huggingface_hub import HfApi

    print(f"Uploading dataset → {dataset_id}")

    def load_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_records = load_jsonl(data_dir / "processed/train.jsonl")
    eval_records  = load_jsonl(data_dir / "processed/eval.jsonl")

    ds = DatasetDict({
        "train": Dataset.from_list(train_records),
        "eval":  Dataset.from_list(eval_records),
    })

    dataset_card = f"""---
language:
  - en
  - luy
license: cc-by-4.0
task_categories:
  - translation
tags:
  - luhya
  - lumarachi
  - lulogooli
  - maragoli
  - kenyan-languages
  - african-languages
  - low-resource
  - translation
---

# English-Luhya Translations (Lumarachi + Lulogooli)

Parallel English-Luhya sentence pairs for training machine translation models.
Covers two Luhya dialects spoken in western Kenya: **Lumarachi** and **Lulogooli (Maragoli)**.

## Data sources

| Source | Pairs | Notes |
|--------|-------|-------|
| KenTrans (Kencorpus, Harvard Dataverse) | ~10,900 | Luhya-Swahili, pivoted to English via NLLB |
| Bible corpus (Chimoto & Bassett, 2022) | ~7,952 | Luhya-Marama / English NT |
| Contributed pairs | varies | Native speaker reviewed |

## Format

```json
{{"english": "Hello, how are you?", "luhya": "...", "dialect": "lumarachi", "source": "contributed"}}
```

## Splits

| Split | Examples |
|-------|---------|
| train | {len(train_records):,} |
| eval  | {len(eval_records):,} |

## Citation

If you use this dataset, please also cite the original Kencorpus:

```bibtex
@article{{wanjawa2023kencorpus,
  title={{Kencorpus: A Kenyan Language Corpus of Swahili, Dholuo and Luhya for NLP Tasks}},
  author={{Wanjawa, Barack and others}},
  journal={{Journal for Language Technology and Computational Linguistics}},
  year={{2023}}
}}
```

## Acknowledgments

- Kencorpus / Maseno University, University of Nairobi, Africa Nazarene University
- Chimoto & Bassett (2022) for the Luhya-Marama Bible corpus
- Lacuna Fund for supporting KenCorpus data collection
"""

    ds.push_to_hub(dataset_id, private=False)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=dataset_id,
        repo_type="dataset",
    )
    print(f"Dataset uploaded → https://huggingface.co/datasets/{dataset_id}")


def generate_model_card(model_id: str, dataset_id: str, output_path: Path, eval_results: dict = None):
    bleu  = eval_results.get("BLEU",   "TBD") if eval_results else "TBD"
    chrf  = eval_results.get("chrF++", "TBD") if eval_results else "TBD"
    today = date.today().isoformat()

    card = f"""---
language:
  - en
  - luy
license: apache-2.0
base_model: google/translategemma-4b-it
tags:
  - translation
  - translategemma
  - luhya
  - lumarachi
  - lulogooli
  - kenyan-languages
  - african-languages
  - low-resource
  - lora
  - rslora
  - unsloth
model-index:
  - name: {model_id}
    results:
      - task:
          type: translation
        dataset:
          name: {dataset_id}
          type: {dataset_id}
        metrics:
          - type: bleu
            value: {bleu}
          - type: chrf
            value: {chrf}
---

# Luhya TranslateGemma-4B V1 (Lumarachi + Lulogooli)

Fine-tuned **English → Luhya** translation model based on Google's
[TranslateGemma-4B-it](https://huggingface.co/google/translategemma-4b-it).

Covers two Luhya dialects: **Lumarachi** and **Lulogooli (Maragoli)**,
spoken by an estimated 2–3 million people in western Kenya.

## Model Details

| Attribute | Value |
|-----------|-------|
| Base model | google/translategemma-4b-it |
| Fine-tuning method | rsLoRA, high-rank |
| LoRA rank / alpha | r=256, alpha=256 |
| Training data | [English-Luhya dataset](https://huggingface.co/datasets/{dataset_id}) |
| Direction | English → Luhya |
| BLEU | **{bleu}** |
| chrF++ | **{chrf}** |
| Framework | Unsloth + TRL |
| Training date | {today} |

## Usage

```python
from unsloth import FastLanguageModel
import torch

model, processor = FastLanguageModel.from_pretrained(
    model_name="{model_id}",
    max_seq_length=512,
)
text_tokenizer = getattr(processor, "tokenizer", processor)
FastLanguageModel.for_inference(model)

def translate_to_luhya(text: str) -> str:
    messages = [{{
        "role": "user",
        "content": [{{
            "type": "text",
            "source_lang_code": "en",
            "target_lang_code": "luy",
            "text": text,
        }}],
    }}]
    formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = text_tokenizer([formatted], return_tensors="pt", padding=True)
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return text_tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

print(translate_to_luhya("Hello, how are you?"))
```

## Training Details

- **Datasets**: KenTrans (Kencorpus), Luhya-Marama Bible corpus, contributed pairs
- **Dialects**: Lumarachi, Lulogooli
- **Pivot strategy**: KenTrans Luhya-Swahili pairs pivoted to English via NLLB-200

## Limitations

- English → Luhya direction only
- Best for general/conversational text; specialized domains need human review
- Dialect coverage weighted toward availability in source corpora

## Acknowledgments

- Google for TranslateGemma-4B-it
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- Kencorpus team (Maseno University, University of Nairobi, Africa Nazarene University)
- [gateremark](https://huggingface.co/gateremark) for the Kikuyu model recipe this is based on

## Citation

```bibtex
@misc{{{model_id.split("/")[-1]}_{today[:4]},
  author = {{YOUR NAME}},
  title = {{Luhya TranslateGemma-4B: English to Luhya Translation}},
  year = {{{today[:4]}}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{model_id}}}}}
}}
```
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(card, encoding="utf-8")
    print(f"Model card written → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id",  required=True, help="HF dataset repo, e.g. yourname/english-luhya-translations")
    parser.add_argument("--model-id",    required=True, help="HF model repo, e.g. yourname/luhya_translategemma_4b_v1")
    parser.add_argument("--data-dir",    default="data",             help="Local data directory")
    parser.add_argument("--eval-results",default="evaluation/results.json", help="Eval results JSON (optional)")
    parser.add_argument("--skip-dataset",action="store_true",         help="Skip dataset upload (model card only)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    eval_results = None
    eval_path = base / args.eval_results
    if eval_path.exists():
        eval_results = json.loads(eval_path.read_text())

    if not args.skip_dataset:
        upload_dataset(args.dataset_id, base / args.data_dir)

    generate_model_card(
        model_id=args.model_id,
        dataset_id=args.dataset_id,
        output_path=base / "upload" / "README.md",
        eval_results=eval_results,
    )
