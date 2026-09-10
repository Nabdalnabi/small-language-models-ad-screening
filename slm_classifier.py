"""Full fine-tuning and separate local evaluation for transcript classification.

Adapted from the supplied DeepSeek training and SmolLM evaluation notebooks.
This is a research code sample, not a diagnostic system.
"""

import argparse
import json
import os
from pathlib import Path

from data_utils import check_splits, encode_labels, label_mapping, read_rows


# Configuration and local-only reporting
def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "evaluate"])
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--revision", default="main", help="Use a commit SHA for reproducibility")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-csv")
    parser.add_argument("--test-csv")
    parser.add_argument("--checkpoint", help="Local checkpoint directory for evaluation")
    parser.add_argument("--output-dir", required=True, help="New or empty local output directory")
    parser.add_argument("--text-col", default="input")
    parser.add_argument("--label-col", default="output")
    parser.add_argument("--group-col", help="Participant ID column for split checks")
    parser.add_argument("--positive-label", help="Exact AD class label for sensitivity/AUC")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Use cached models only")
    args = parser.parse_args()
    if min(args.max_length, args.batch_size, args.gradient_accumulation, args.epochs, args.patience) < 1:
        parser.error("Lengths, batch sizes, epochs, and patience must be positive.")
    if args.learning_rate <= 0 or args.weight_decay < 0 or not 0 <= args.warmup_ratio <= 1:
        parser.error("Invalid optimization settings.")
    if args.mode == "train" and not (args.train_csv and args.val_csv):
        parser.error("Training requires --train-csv and --val-csv.")
    if args.mode == "evaluate" and not (args.checkpoint and args.test_csv):
        parser.error("Evaluation requires --checkpoint and --test-csv.")
    return args


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    args = arguments()
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # Heavy imports are deferred so help and data checks require no model downloads.
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, classification_report,
        confusion_matrix, f1_score, roc_auc_score,
    )
    from transformers import (
        AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
        EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
    )

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be empty to avoid overwriting a run.")

    # Training never reads the held-out test set. Check it against train/validation
    # during final evaluation by passing the original CSV paths again.
    split_paths = ({"train": args.train_csv, "validation": args.val_csv}
                   if args.mode == "train" else
                   {"test": args.test_csv, "train": args.train_csv, "validation": args.val_csv})
    rows = {name: read_rows(path, args.text_col, args.label_col, args.group_col)
            for name, path in split_paths.items() if path}
    check_splits(rows, grouped=bool(args.group_col))
    print("Participant separation checked." if args.group_col else
          "Participant separation unverified: no --group-col supplied.")

    if args.mode == "train":
        mapping = label_mapping(rows["train"])
        positive_label = args.positive_label
        model_source = args.model
        load_options = {"revision": args.revision, "local_files_only": args.offline}
        max_length = args.max_length
    else:
        model_source = str(Path(args.checkpoint).resolve(strict=True))
        metadata = json.loads((Path(model_source) / "experiment.json").read_text())
        mapping = metadata["label2id"]
        positive_label = metadata["positive_label"]
        if args.positive_label is not None and args.positive_label != positive_label:
            raise ValueError("Positive label differs from the saved experiment.")
        max_length = metadata["max_length"]
        load_options = {"local_files_only": True}

    if positive_label is not None and positive_label not in mapping:
        raise ValueError("Positive label must be present in the training class mapping.")
    for split in rows.values():
        encode_labels(split, mapping)

    # Use the model's own tokenizer and pass masks during training AND inference.
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=False, **load_options)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS or padding token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, num_labels=2, label2id=mapping,
        id2label={v: k for k, v in mapping.items()},
        pad_token_id=tokenizer.pad_token_id, trust_remote_code=False, **load_options,
    )
    model.config.use_cache = False
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context_limit, int) and max_length > context_limit:
        raise ValueError("Requested maximum length exceeds model context length.")

    class TranscriptDataset(torch.utils.data.Dataset):
        def __init__(self, records):
            self.encodings = tokenizer([r["text"] for r in records], truncation=True,
                                       max_length=max_length, padding=False)
            self.labels = encode_labels(records, mapping)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {**{k: v[index] for k, v in self.encodings.items()},
                    "labels": self.labels[index]}

    def metrics(prediction):
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        labels = prediction.label_ids
        predicted = logits.argmax(axis=-1)
        result = {
            "accuracy": float(accuracy_score(labels, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
            "f1_micro": float(f1_score(labels, predicted, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(labels, predicted, labels=[0, 1], average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        }
        if positive_label is not None and len(np.unique(labels)) == 2:
            positive_id = mapping[positive_label]
            probabilities = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
            result["roc_auc"] = float(roc_auc_score(labels == positive_id, probabilities[:, positive_id]))
        return result

    datasets = {k: TranscriptDataset(v) for k, v in rows.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available() and not args.cpu
    training_args = TrainingArguments(
        output_dir=str(output_dir), use_cpu=args.cpu,
        num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=args.mode == "train", max_grad_norm=1.0,
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        eval_strategy="epoch" if args.mode == "train" else "no",
        save_strategy="epoch" if args.mode == "train" else "no",
        load_best_model_at_end=args.mode == "train", metric_for_best_model="f1_macro",
        greater_is_better=True, save_total_limit=2,
        report_to="none", push_to_hub=False, seed=args.seed, data_seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args, processing_class=tokenizer,
        train_dataset=datasets.get("train") if args.mode == "train" else None,
        eval_dataset=datasets.get("validation") if args.mode == "train" else None,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)]
        if args.mode == "train" else None,
    )

    if args.mode == "train":
        # Each invocation starts from pretrained weights, never a prior trial's model.
        trainer.train()
        checkpoint = output_dir / "best_model"
        trainer.save_model(str(checkpoint))
        tokenizer.save_pretrained(checkpoint)
        write_json(checkpoint / "experiment.json", {
            "model": args.model, "requested_revision": args.revision,
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "label2id": mapping, "positive_label": positive_label,
            "max_length": max_length, "seed": args.seed,
            "selection_metric": "validation_f1_macro",
            "participant_separation_checked": bool(args.group_col),
            "parameters": sum(p.numel() for p in model.parameters()),
        })
        result = trainer.evaluate()
        write_json(output_dir / "validation_metrics.json", result)
    else:
        prediction = trainer.predict(datasets["test"])
        logits = prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predicted = logits.argmax(axis=-1)
        names = [name for name, i in sorted(mapping.items(), key=lambda item: item[1])]
        cm = confusion_matrix(prediction.label_ids, predicted, labels=[0, 1])
        result = {**prediction.metrics, "class_order": names, "confusion_matrix": cm.tolist(),
                  "classification_report": classification_report(
                      prediction.label_ids, predicted, labels=[0, 1], target_names=names,
                      zero_division=0, output_dict=True),
                  "overlap_check_against_train_and_validation": bool(args.train_csv and args.val_csv),
                  "participant_ids_supplied": bool(args.group_col)}
        if positive_label is not None:
            pos = mapping[positive_label]
            neg = 1 - pos
            result["sensitivity"] = float(cm[pos, pos] / cm[pos].sum()) if cm[pos].sum() else None
            result["specificity"] = float(cm[neg, neg] / cm[neg].sum()) if cm[neg].sum() else None
        write_json(output_dir / "test_metrics.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
