"""
05_evaluate.py
==============
Evaluate the trained Luhya TranslateGemma model.
Outputs BLEU, chrF++, eval loss — same metrics as Kikuyu V7.

Usage:
    python scripts/05_evaluate.py --model-dir training/final_lora
    python scripts/05_evaluate.py --model-id yourname/luhya_translategemma_4b_v1
"""

import json
import argparse
from pathlib import Path


def load_eval_pairs(eval_file: str) -> list[dict]:
    pairs = []
    with open(eval_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))
    return pairs


def run_evaluation(
    model_dir_or_id: str,
    eval_file: str,
    output_file: str,
    n_samples: int = None,
    src_lang: str = "en",
    tgt_lang: str = "luy",
):
    import torch
    from unsloth import FastLanguageModel
    import sacrebleu

    print(f"\n=== Luhya TranslateGemma — Evaluation ===")
    print(f"  Model : {model_dir_or_id}")
    print(f"  Eval  : {eval_file}")

    # Load model
    model, processor = FastLanguageModel.from_pretrained(
        model_name=model_dir_or_id,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    text_tokenizer = (
        getattr(processor, "tokenizer", None)
        or getattr(processor, "text_tokenizer", None)
        or processor
    )
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    model.config.pad_token_id = text_tokenizer.pad_token_id
    text_tokenizer.padding_side = "left"

    terminators = []
    for tok in [
        text_tokenizer.eos_token_id,
        text_tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        text_tokenizer.convert_tokens_to_ids("<eos>"),
    ]:
        if isinstance(tok, int) and tok >= 0 and tok not in terminators:
            terminators.append(tok)

    # Load eval pairs
    pairs = load_eval_pairs(eval_file)
    if n_samples:
        import random; random.seed(42)
        pairs = random.sample(pairs, min(n_samples, len(pairs)))
    print(f"  Evaluating on {len(pairs)} pairs...\n")

    references, hypotheses, sample_output = [], [], []

    for i, pair in enumerate(pairs):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "source_lang_code": src_lang,
                        "target_lang_code": tgt_lang,
                        "text": pair["english"],
                    }
                ],
            }
        ]
        formatted = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = text_tokenizer([formatted], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=text_tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        pred = text_tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        ).strip()

        references.append(pair["luhya"])
        hypotheses.append(pred)

        if i < 10:
            sample_output.append({
                "english":   pair["english"],
                "reference": pair["luhya"],
                "predicted": pred,
                "dialect":   pair.get("dialect", "?"),
            })

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(pairs)}")

    # Compute metrics
    bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf_score = sacrebleu.corpus_chrf(hypotheses, [references], beta=2)

    results = {
        "model": model_dir_or_id,
        "eval_pairs": len(pairs),
        "BLEU": round(bleu_score.score, 4),
        "chrF++": round(chrf_score.score, 4),
        "samples": sample_output,
    }

    print(f"\n{'='*40}")
    print(f"  BLEU    : {results['BLEU']}")
    print(f"  chrF++  : {results['chrF++']}")
    print(f"{'='*40}\n")

    print("Sample translations:")
    for s in sample_output[:5]:
        print(f"  EN  : {s['english']}")
        print(f"  REF : {s['reference']}  [{s['dialect']}]")
        print(f"  PRED: {s['predicted']}")
        print()

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved → {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",  default=None, help="Local path to merged model or LoRA adapter")
    parser.add_argument("--model-id",   default=None, help="HuggingFace Hub model ID")
    parser.add_argument("--eval-file",  default="data/processed/eval.jsonl")
    parser.add_argument("--output",     default="evaluation/results.json")
    parser.add_argument("--n-samples",  type=int, default=None, help="Limit eval set size")
    args = parser.parse_args()

    model_ref = args.model_dir or args.model_id
    if not model_ref:
        raise ValueError("Provide --model-dir or --model-id")

    base = Path(__file__).parent.parent
    run_evaluation(
        model_dir_or_id=model_ref,
        eval_file=str(base / args.eval_file),
        output_file=str(base / args.output),
        n_samples=args.n_samples,
    )
