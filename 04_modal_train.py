"""
04_modal_train.py
=================
Run training on Modal with an H100 — same infra the Kikuyu model used.

Setup:
    pip install modal
    modal setup        # authenticate once
    modal run scripts/04_modal_train.py

To push to HuggingFace Hub after training, add your token:
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal run scripts/04_modal_train.py --push-hub yourname/luhya_translategemma_4b_v1
"""

import os
import sys

try:
    import modal
except ImportError:
    print("Modal not installed. Run: pip install modal && modal setup")
    sys.exit(1)

# ── Modal image with all dependencies ───────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "trl>=0.8.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "accelerate>=0.29.0",
        "bitsandbytes>=0.43.0",
        "sentencepiece",
        "sacrebleu",
        "evaluate",
    )
    .run_commands(
        "pip install flash-attn --no-build-isolation",
    )
)

app = modal.App("luhya-translategemma", image=image)

# Mount local project directory into the container
project_mount = modal.Mount.from_local_dir(
    local_path=".",
    remote_path="/root/luhya_translate",
    condition=lambda p: not any(
        seg in p for seg in [".git", "__pycache__", ".venv", "*.pyc"]
    ),
)

hf_secret = modal.Secret.from_name("huggingface", required=False)


@app.function(
    gpu="H100",                    # same as Kikuyu V7
    timeout=60 * 60 * 8,          # 8 hours max
    mounts=[project_mount],
    secrets=[hf_secret] if hf_secret else [],
    volumes={"/root/outputs": modal.Volume.from_name("luhya-outputs", create_if_missing=True)},
)
def run_training(
    hub_model_id: str = None,
    smoke_test: bool = False,
    epochs: int = 5,
    rank: int = 256,
):
    import subprocess, sys

    os.chdir("/root/luhya_translate")
    sys.path.insert(0, "/root/luhya_translate/scripts")

    cmd = [
        sys.executable, "scripts/03_train.py",
        "--local",
        f"--epochs={epochs}",
        f"--rank={rank}",
    ]
    if hub_model_id:
        cmd.append(f"--hub-id={hub_model_id}")
    if smoke_test:
        cmd.append("--smoke-test")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("Training failed")

    # Copy outputs to persistent volume
    subprocess.run(["cp", "-r", "training/final_lora", "/root/outputs/final_lora"])
    print("Saved to Modal Volume /root/outputs/final_lora")


@app.local_entrypoint()
def main(
    hub_id: str = None,
    smoke_test: bool = False,
    epochs: int = 5,
    rank: int = 256,
):
    print(f"Launching Luhya TranslateGemma training on Modal H100...")
    print(f"  Epochs: {epochs}, LoRA rank: {rank}")
    if hub_id:
        print(f"  Will push to: {hub_id}")
    if smoke_test:
        print("  ⚠ SMOKE TEST MODE — tiny data, sanity check only")

    run_training.remote(
        hub_model_id=hub_id,
        smoke_test=smoke_test,
        epochs=epochs,
        rank=rank,
    )
    print("Done! Check Modal dashboard for logs.")
