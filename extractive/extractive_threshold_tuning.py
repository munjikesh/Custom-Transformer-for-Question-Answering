import argparse
import collections
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from transformers import AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from extractive_finetuning import BertForQuestionAnswering
from mlm_pretraining import ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_checkpoint_dir", required=True, help="Dir with model_config.json")
    p.add_argument("--finetuned_model_dir", required=True, help="Dir with model.safetensors")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--threshold_points", type=int, default=201)
    p.add_argument("--out_json", default="squad_v2_threshold_tuning.json")
    return p.parse_args()


def main():
    args = parse_args()
    pretrain_dir = Path(args.pretrain_checkpoint_dir)
    finetuned_dir = Path(args.finetuned_model_dir)

    cfg = ModelConfig(**json.loads((pretrain_dir / "model_config.json").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(str(finetuned_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token

    model = BertForQuestionAnswering(cfg)
    state = load_file(str(finetuned_dir / "model.safetensors"))
    model.load_state_dict(state, strict=True)

    data = load_dataset("squad_v2")["validation"]
    max_length = min(args.max_length, cfg.max_position_embeddings)
    doc_stride = min(args.doc_stride, max(8, max_length // 4))

    def prep(examples):
        tok = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            sids = tok.sequence_ids(i)
            ex_i = sample_map[i]
            tok["example_id"].append(examples["id"][ex_i])
            tok["offset_mapping"][i] = [o if sids[k] == 1 else None for k, o in enumerate(tok["offset_mapping"][i])]
        return tok

    eval_features = data.map(prep, batched=True, remove_columns=data.column_names)
    eval_features_model = eval_features.remove_columns(["example_id", "offset_mapping"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="tmp_eval_squadv2_threshold",
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=args.num_workers,
            report_to="none",
            remove_unused_columns=False,
        ),
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        tokenizer=tokenizer,
    )

    preds, _, _ = trainer.predict(eval_features_model)
    start_logits, end_logits = preds

    ex_id_to_idx = {k: i for i, k in enumerate(data["id"])}
    feat_per_ex = collections.defaultdict(list)
    for i, f in enumerate(eval_features):
        feat_per_ex[ex_id_to_idx[f["example_id"]]].append(i)

    n_best = 20
    max_answer_length = 30
    best_span_text = {}
    score_diff = {}
    references = []

    for ex_idx, ex in enumerate(data):
        context = ex["context"]
        best_score = -1e30
        best_text = ""
        best_null = 1e30
        for fi in feat_per_ex[ex_idx]:
            sl = start_logits[fi]
            el = end_logits[fi]
            offs = eval_features[fi]["offset_mapping"]
            cls_idx = eval_features[fi]["input_ids"].index(tokenizer.cls_token_id)
            null_score = float(sl[cls_idx] + el[cls_idx])
            if null_score < best_null:
                best_null = null_score

            s_idx = np.argsort(sl)[-1 : -n_best - 1 : -1].tolist()
            e_idx = np.argsort(el)[-1 : -n_best - 1 : -1].tolist()
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offs) or e >= len(offs):
                        continue
                    if offs[s] is None or offs[e] is None:
                        continue
                    if e < s or (e - s + 1) > max_answer_length:
                        continue
                    st, en = offs[s][0], offs[e][1]
                    sc = float(sl[s] + el[e])
                    if sc > best_score:
                        best_score = sc
                        best_text = context[st:en]

        best_span_text[ex["id"]] = best_text
        score_diff[ex["id"]] = best_null - best_score
        references.append({"id": ex["id"], "answers": ex["answers"]})

    metric = evaluate.load("squad_v2")
    diffs = np.array(list(score_diff.values()), dtype=np.float32)
    lo, hi = float(diffs.min()), float(diffs.max())
    thresholds = np.linspace(lo, hi, num=max(3, args.threshold_points)).tolist()
    if 0.0 not in thresholds:
        thresholds.append(0.0)
    thresholds = sorted(set(round(t, 6) for t in thresholds))

    best = {"f1": -1.0, "exact": -1.0, "threshold": 0.0}
    for th in thresholds:
        preds = []
        for ex in data:
            eid = ex["id"]
            text = "" if score_diff[eid] > th else best_span_text[eid]
            preds.append({"id": eid, "prediction_text": text, "no_answer_probability": 0.0})
        res = metric.compute(predictions=preds, references=references)
        if res["f1"] > best["f1"]:
            best = {"f1": res["f1"], "exact": res["exact"], "threshold": th}

    zero_preds = []
    for ex in data:
        eid = ex["id"]
        text = "" if score_diff[eid] > 0.0 else best_span_text[eid]
        zero_preds.append({"id": eid, "prediction_text": text, "no_answer_probability": 0.0})
    zero_res = metric.compute(predictions=zero_preds, references=references)

    out = {
        "model_dir": str(finetuned_dir),
        "threshold_zero": zero_res,
        "best_threshold": best["threshold"],
        "best_threshold_metrics": {"exact": best["exact"], "f1": best["f1"]},
    }

    out_path = finetuned_dir / args.out_json
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Saved threshold tuning report: {out_path}")


if __name__ == "__main__":
    main()
