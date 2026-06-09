"""
03_train.py
===========
Luhya TranslateGemma — Training script
Based on gateremark/kikuyu_translategemma_4b_v7_highrank_rslora recipe.

Model   : google/translategemma-4b-it
Method  : rsLoRA, rank=256, alpha=256
Target  : English → Luhya (Lumarachi + Lulogooli)
Lang code: luy  (ISO 639-3 for Luhya)
Platform: Unsloth + TRL + Transformers
GPU     : H100 recommended (via Modal or Colab A100 fallback)

Usage:
    # On Modal (recommended):
    modal run scripts/03_train.py

    # Local (GPU required, 16GB+ VRAM):
    python scripts/03_train.py --local

    # Quick smoke-test (CPU, tiny data):
    python scripts/03_train.py --local --smoke-test
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ── config ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # Model
    base_model: str = "google/translategemma-4b-it"
    output_dir: str = "training/checkpoints"
    hub_model_id: Optional[str] = None          # set to "yourname/luhya_translategemma_4b_v1"

    # Data
    train_file: str = "data/augmented/train_augmented.jsonl"
    eval_file:  str = "data/processed/eval.jsonl"

    # LoRA — mirror of Kikuyu V7
    lora_rank:  int = 256
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    use_rslora: bool = True
    use_dora:   bool = False
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # Training
    max_seq_length: int = 512       # TranslateGemma prompt is short
    num_train_epochs: int = 5       # +2 vs Kikuyu V7 to compensate smaller dataset
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8   # effective batch = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    optim: str = "adamw_8bit"
    fp16: bool = False
    bf16: bool = True

    # Eval
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 20
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Language
    src_lang: str = "en"
    tgt_lang: str = "luy"   # ISO 639-3 for Luhya

    # Inference
    max_new_tokens: int = 256

    # Smoke test
    smoke_test: bool = False


CFG = TrainConfig()


# ── prompt template ─────────────────────────────────────────────────────────
# Identical to Kikuyu model — just lang codes change.
def make_messages(english: str, luhya: str = None) -> list[dict]:
    """
    TranslateGemma chat template format.
    If luhya is None, generates inference-only messages (no assistant turn).
    """
    user_content = [
        {
            "type": "text",
            "source_lang_code": CFG.src_lang,
            "target_lang_code": CFG.tgt_lang,
            "text": english,
        }
    ]
    messages = [{"role": "user", "content": user_content}]
    if luhya is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": luhya}]})
    return messages


def format_example(record: dict, processor) -> str:
    """Format a training example using the processor's chat template."""
    messages = make_messages(record["english"], record["luhya"])
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


# ── dataset loading ─────────────────────────────────────────────────────────
def load_hf_dataset(train_file: str, eval_file: str, smoke_test: bool = False):
    """Load JSONL into HuggingFace Dataset objects."""
    from datasets import load_dataset, Dataset
    import json

    def read_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    train_records = read_jsonl(train_file)
    eval_records  = read_jsonl(eval_file)

    if smoke_test:
        train_records = train_records[:64]
        eval_records  = eval_records[:16]
        print(f"[smoke-test] Using {len(train_records)} train / {len(eval_records)} eval pairs")

    train_ds = Dataset.from_list(train_records)
    eval_ds  = Dataset.from_list(eval_records)
    return train_ds, eval_ds


# ── main training function ──────────────────────────────────────────────────
def train(cfg: TrainConfig):
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from transformers import EarlyStoppingCallback

    print("\n=== Luhya TranslateGemma — Training ===")
    print(f"  Base model : {cfg.base_model}")
    print(f"  LoRA       : r={cfg.lora_rank}, alpha={cfg.lora_alpha}, rsLoRA={cfg.use_rslora}")
    print(f"  Epochs     : {cfg.num_train_epochs}")
    print(f"  Lang codes : {cfg.src_lang} → {cfg.tgt_lang}\n")

    # ── load base model ──
    model, processor = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,          # auto-detect bf16
        load_in_4bit=False,  # full bf16 for quality; set True if < 16GB VRAM
    )

    # ── attach LoRA adapter ──
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
        bias="none",
        random_state=42,
    )

    # ── tokenizer setup ──
    text_tokenizer = (
        getattr(processor, "tokenizer", None)
        or getattr(processor, "text_tokenizer", None)
        or processor
    )
    if text_tokenizer.pad_token_id is None:
        text_tokenizer.pad_token = text_tokenizer.eos_token
    model.config.pad_token_id = text_tokenizer.pad_token_id
    text_tokenizer.padding_side = "right"   # right padding during training

    # ── datasets ──
    base = Path(__file__).parent.parent
    train_ds, eval_ds = load_hf_dataset(
        str(base / cfg.train_file),
        str(base / cfg.eval_file),
        smoke_test=cfg.smoke_test,
    )

    def format_fn(examples):
        texts = []
        for en, lh in zip(examples["english"], examples["luhya"]):
            record = {"english": en, "luhya": lh}
            texts.append(format_example(record, processor))
        return {"text": texts}

    train_ds = train_ds.map(format_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(format_fn, batched=True, remove_columns=eval_ds.column_names)

    # ── training args ──
    output_dir = str(base / cfg.output_dir)
    sft_cfg = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        eval_strategy=cfg.eval_strategy,
        eval_steps=cfg.eval_steps if not cfg.smoke_test else 10,
        save_steps=cfg.save_steps if not cfg.smoke_test else 10,
        logging_steps=cfg.logging_steps,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model,
        save_total_limit=3,
        report_to="none",          # set "wandb" if you want W&B tracking
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=text_tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=sft_cfg,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # ── train ──
    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"\nTraining complete. Runtime: {trainer_stats.metrics['train_runtime']:.0f}s")

    # ── save ──
    final_dir = base / "training" / "final_lora"
    model.save_pretrained(str(final_dir))
    text_tokenizer.save_pretrained(str(final_dir))
    print(f"LoRA adapter saved → {final_dir}")

    # ── optionally push to Hub ──
    if cfg.hub_model_id:
        print(f"Pushing to Hub: {cfg.hub_model_id}")
        model.push_to_hub(cfg.hub_model_id)
        text_tokenizer.push_to_hub(cfg.hub_model_id)
        print("Done! ✓")

    return model, text_tokenizer, processor


# ── quick inference test ────────────────────────────────────────────────────
def translate(text: str, model, text_tokenizer, processor, cfg: TrainConfig) -> str:
    import torch
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    terminators = []
    for tok in [
        text_tokenizer.eos_token_id,
        text_tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        text_tokenizer.convert_tokens_to_ids("<eos>"),
    ]:
        if isinstance(tok, int) and tok >= 0 and tok not in terminators:
            terminators.append(tok)

    messages = make_messages(text)
    formatted = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = text_tokenizer([formatted], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,
            eos_token_id=terminators,
            pad_token_id=text_tokenizer.pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    return text_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


# ── entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local",       action="store_true", help="Run locally (not via Modal)")
    parser.add_argument("--smoke-test",  action="store_true", help="Tiny run for debugging")
    parser.add_argument("--hub-id",      default=None,        help="HuggingFace Hub model ID to push to")
    parser.add_argument("--epochs",      type=int, default=5)
    parser.add_argument("--rank",        type=int, default=256)
    parser.add_argument("--train-file",  default="data/augmented/train_augmented.jsonl")
    parser.add_argument("--eval-file",   default="data/processed/eval.jsonl")
    args = parser.parse_args()

    CFG.num_train_epochs = args.epochs
    CFG.lora_rank        = args.rank
    CFG.hub_model_id     = args.hub_id
    CFG.smoke_test       = args.smoke_test
    CFG.train_file       = args.train_file
    CFG.eval_file        = args.eval_file

    model, text_tokenizer, processor = train(CFG)

    # Quick sanity check
    print("\n--- Inference sanity check ---")
    test_sentences = [
        "Hello, how are you?",
        "The rain is coming today.",
        "I want to go to the market.",
        "My name is Tanga.",
    ]
    for sent in test_sentences:
        out = translate(sent, model, text_tokenizer, processor, CFG)
        print(f"  EN : {sent}")
        print(f"  LUY: {out}\n")
