from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from script.common.imcs21_common import BIO_LABELS, entity_prf, sha256


ROOT = Path(__file__).resolve().parents[2]
BASE_MODEL = ROOT / "resources" / "models" / "roberta-med-inquiry-base"
MODEL_OUTPUT = ROOT / "resources" / "models" / "imcs21-roberta-ner"
RUN_OUTPUT = ROOT / "outputs" / "01_evidence_extractor_training"
LABEL2ID = {label: index for index, label in enumerate(BIO_LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


class EncodedSentences(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int) -> None:
        self.items = []
        for offset in range(0, len(rows), 4096):
            batch = rows[offset:offset + 4096]
            encoded = tokenizer(
                [row["text"] for row in batch],
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
            )
            for index, row in enumerate(batch):
                labels = row["labels"]
                token_labels = []
                for start, end in encoded["offset_mapping"][index]:
                    token_labels.append(-100 if end <= start else LABEL2ID[labels[start]])
                self.items.append({
                    "input_ids": encoded["input_ids"][index],
                    "attention_mask": encoded["attention_mask"][index],
                    "token_type_ids": encoded.get("token_type_ids", [None] * len(batch))[index],
                    "labels": token_labels,
                })
        for item in self.items:
            if item["token_type_ids"] is None:
                item.pop("token_type_ids")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def load_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        {"text": turn["sentence"], "labels": turn["BIO_label"].split()}
        for sample in data.values()
        for turn in sample["dialogue"]
    ]


def spans_from_token_ids(sequence: list[int]) -> list[str]:
    return [ID2LABEL[index] for index in sequence if index != -100]


def metrics(eval_prediction) -> dict[str, float]:
    logits, labels = eval_prediction
    predictions = np.argmax(logits, axis=-1)
    gold_sequences = []
    predicted_sequences = []
    for gold, predicted in zip(labels, predictions):
        mask = gold != -100
        gold_sequences.append(spans_from_token_ids(gold[mask].tolist()))
        predicted_sequences.append(spans_from_token_ids(predicted[mask].tolist()))
    score = entity_prf(gold_sequences, predicted_sequences)
    return {"entity_precision": score["precision"], "entity_recall": score["recall"], "entity_f1": score["f1"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-dev", type=int)
    args = parser.parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    MODEL_OUTPUT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT.mkdir(parents=True, exist_ok=True)

    train_rows = load_rows(ROOT / "train.json")
    dev_rows = load_rows(ROOT / "dev.json")
    if args.limit_train:
        train_rows = train_rows[:args.limit_train]
    if args.limit_dev:
        dev_rows = dev_rows[:args.limit_dev]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True, use_fast=True)
    train_dataset = EncodedSentences(train_rows, tokenizer, args.max_length)
    dev_dataset = EncodedSentences(dev_rows, tokenizer, args.max_length)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        local_files_only=True,
        num_labels=len(BIO_LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        ignore_mismatched_sizes=True,
    )
    training_args = TrainingArguments(
        output_dir=str(RUN_OUTPUT / "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        group_by_length=True,
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    evaluation = trainer.evaluate()
    trainer.save_model(MODEL_OUTPUT)
    tokenizer.save_pretrained(MODEL_OUTPUT)
    manifest = {
        "schema_version": "imcs21_roberta_ner_training_v1",
        "base_model": str(BASE_MODEL),
        "train_sentences": len(train_dataset),
        "dev_sentences": len(dev_dataset),
        "labels": list(BIO_LABELS),
        "hyperparameters": vars(args),
        "evaluation": evaluation,
        "source_hashes": {"train": sha256(ROOT / "train.json"), "dev": sha256(ROOT / "dev.json")},
    }
    (RUN_OUTPUT / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
