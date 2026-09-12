from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get(
    "EVIDENCE_STATE_STUDENT_OUTPUT",
    ROOT / "outputs" / "student_distillation",
))
MODEL = os.environ.get("QWEN3_MODEL_PATH", str(ROOT / "resources/models/Qwen3-4B"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = DATA / ("qwen3_4b_qlora_smoke" if args.smoke else "qwen3_4b_qlora_v1")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def encode(row: dict) -> dict:
        prefix = tokenizer.apply_chat_template(
            row["messages"][:2], tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )["input_ids"]
        full = tokenizer.apply_chat_template(
            row["messages"], tokenize=True, add_generation_prompt=False,
            enable_thinking=False,
        )["input_ids"]
        return {"input_ids": full, "attention_mask": [1] * len(full), "labels": [-100] * len(prefix) + full[len(prefix):]}

    train_rows = read_jsonl(DATA / "train_silver_sft.jsonl")
    dev_rows = read_jsonl(DATA / "dev_silver_sft.jsonl")
    if args.smoke:
        train_rows, dev_rows = train_rows[:16], dev_rows[:16]
    train = Dataset.from_list(train_rows).map(encode, remove_columns=list(train_rows[0]))
    dev = Dataset.from_list(dev_rows).map(encode, remove_columns=list(dev_rows[0]))

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, quantization_config=quant,
        dtype=torch.bfloat16, device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
    ))
    model.config.use_cache = False
    model.print_trainable_parameters()
    training = TrainingArguments(
        output_dir=str(output), num_train_epochs=2,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=16, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_steps=14,
        bf16=True, gradient_checkpointing=True,
        logging_steps=5, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", optim="paged_adamw_8bit", seed=20260831,
        max_steps=2 if args.smoke else -1,
    )
    trainer = Trainer(
        model=model, args=training, train_dataset=train, eval_dataset=dev,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    trainer.save_model(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    (output / "train_summary.json").write_text(json.dumps({
        "train_rows": len(train), "dev_rows": len(dev),
        "train_metrics": result.metrics, "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
