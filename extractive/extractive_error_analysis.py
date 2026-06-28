import argparse
import collections
import json
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

from extractive_finetuning import BertForQuestionAnswering, load_pretrained_encoder
from mlm_pretraining import ModelConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--dataset", default="squad", choices=["squad", "squad_v2"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--doc_stride", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--out_json", default="qa_error_analysis.json")
    return p.parse_args()


def qtype(question: str) -> str:
    q = question.strip().lower()
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("who"):
        return "who"
    if q.startswith("why"):
        return "why"
    return "other"


def main():
    args = parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    cfg = ModelConfig(**json.loads((ckpt_dir / "model_config.json").read_text(encoding="utf-8")))
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True)
    model = BertForQuestionAnswering(cfg)
    load_pretrained_encoder(model, str(ckpt_dir))

    data = load_dataset(args.dataset)["validation"]
    doc_stride = min(args.doc_stride, max(8, args.max_length // 4))

    def prep(examples):
        tok = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=args.max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
        )
        sm = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            sid = tok.sequence_ids(i)
            ex_i = sm[i]
            tok["example_id"].append(examples["id"][ex_i])
            tok["offset_mapping"][i] = [o if sid[k] == 1 else None for k, o in enumerate(tok["offset_mapping"][i])]
        return tok

    feats = data.map(prep, batched=True, remove_columns=data.column_names)
    feats_model = feats.remove_columns(["example_id", "offset_mapping"])

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="tmp_eval",
            per_device_eval_batch_size=args.batch_size,
            dataloader_num_workers=2,
            report_to="none",
        ),
        tokenizer=tokenizer,
    )
    preds, _, _ = trainer.predict(feats_model)
    s_logits, e_logits = preds

    ex_id_to_idx = {k: i for i, k in enumerate(data["id"])}
    feat_per_ex = collections.defaultdict(list)
    for i, f in enumerate(feats):
        feat_per_ex[ex_id_to_idx[f["example_id"]]].append(i)

    final_preds = {}
    for ex_idx, ex in enumerate(data):
        valid = []
        for fi in feat_per_ex[ex_idx]:
            sl = s_logits[fi]
            el = e_logits[fi]
            offs = feats[fi]["offset_mapping"]
            starts = np.argsort(sl)[-20:][::-1]
            ends = np.argsort(el)[-20:][::-1]
            for s in starts:
                for e in ends:
                    if s >= len(offs) or e >= len(offs) or offs[s] is None or offs[e] is None:
                        continue
                    if e < s or (e - s + 1) > 30:
                        continue
                    st, en = offs[s][0], offs[e][1]
                    valid.append((sl[s] + el[e], ex["context"][st:en]))
        final_preds[ex["id"]] = max(valid, key=lambda x: x[0])[1] if valid else ""

    metric = evaluate.load("squad_v2" if args.dataset == "squad_v2" else "squad")
    by_type = collections.defaultdict(lambda: {"pred": [], "ref": []})
    for ex in data:
        t = qtype(ex["question"])
        pred = {"id": ex["id"], "prediction_text": final_preds[ex["id"]]}
        if args.dataset == "squad_v2":
            pred["no_answer_probability"] = 0.0
        by_type[t]["pred"].append(pred)
        by_type[t]["ref"].append({"id": ex["id"], "answers": ex["answers"]})

    out = {}
    for t, v in by_type.items():
        out[t] = metric.compute(predictions=v["pred"], references=v["ref"])
        out[t]["count"] = len(v["pred"])

    Path(args.out_json).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved error analysis to {args.out_json}")


if __name__ == "__main__":
    main()
