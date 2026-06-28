import argparse
import collections
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import inspect

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, DataCollatorWithPadding

from extractive_finetuning import BertForQuestionAnswering
from mlm_pretraining import ModelConfig


DEFAULT_HF_MODELS = [
    "deepset/roberta-base-squad2",
    "distilbert-base-uncased-distilled-squad",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
]


@dataclass
class ModelRunSpec:
    name: str
    source: str  # "custom" or "hf"
    model_ref: str
    tokenizer_ref: str
    config_ref: Optional[str] = None


def parse_args():
    p = argparse.ArgumentParser(description="Compare extractive QA models on SQuAD v2 validation.")
    p.add_argument(
        "--custom_model_dir",
        default="checkpoints_qa_squad_v2_lr5e-5_len256_e3",
        help="Directory containing model.safetensors and tokenizer files for custom QA model",
    )
    p.add_argument(
        "--custom_config_dir",
        default="checkpoints_pretrain_base_seq256/step_20000",
        help="Directory containing model_config.json for custom QA model",
    )
    p.add_argument(
        "--hf_models",
        default=",".join(DEFAULT_HF_MODELS),
        help="Comma-separated Hugging Face QA model IDs",
    )
    p.add_argument("--skip_custom", action="store_true")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--doc_stride", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--n_best", type=int, default=20)
    p.add_argument("--max_answer_length", type=int, default=30)
    p.add_argument("--max_eval_examples", type=int, default=0, help="0 means full validation split")
    p.add_argument("--output_json", default="comparison_squadv2_results.json")
    p.add_argument("--output_csv", default="comparison_squadv2_results.csv")
    p.add_argument("--review_txt", default="comparison_squadv2_review.txt")
    return p.parse_args()


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def load_custom_model_and_tokenizer(model_dir: Path, config_dir: Path, device: str):
    cfg_path = model_dir / "model_config.json"
    if not cfg_path.exists():
        cfg_path = config_dir / "model_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"model_config.json not found in {model_dir} or {config_dir}")

    cfg = ModelConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.sep_token

    model = BertForQuestionAnswering(cfg)
    state_path = model_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in {model_dir}")
    state = load_file(str(state_path))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model, tok


def load_hf_model_and_tokenizer(model_id: str, device: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tok


def build_eval_dataset(max_eval_examples: int):
    ds = load_dataset("squad_v2", split="validation")
    if max_eval_examples > 0:
        ds = ds.select(range(min(max_eval_examples, len(ds))))
    return ds


def tokenize_with_overflow(dataset, tokenizer, max_length: int, doc_stride: int):
    doc_stride = min(doc_stride, max(8, max_length // 4))

    def prep(examples):
        tok = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        sample_map = tok.pop("overflow_to_sample_mapping")
        tok["example_id"] = []
        for i in range(len(tok["input_ids"])):
            seq_ids = tok.sequence_ids(i)
            ex_i = sample_map[i]
            tok["example_id"].append(examples["id"][ex_i])
            tok["offset_mapping"][i] = [o if seq_ids[k] == 1 else None for k, o in enumerate(tok["offset_mapping"][i])]
        return tok

    feats = dataset.map(prep, batched=True, remove_columns=dataset.column_names)
    return feats


def collect_logits(
    model,
    tokenizer,
    features,
    batch_size: int,
    device: str,
):
    model_inputs = features.remove_columns(["example_id", "offset_mapping"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loader = DataLoader(model_inputs, batch_size=batch_size, shuffle=False, collate_fn=collator)

    accepts_token_type_ids = "token_type_ids" in inspect.signature(model.forward).parameters

    start_chunks = []
    end_chunks = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if accepts_token_type_ids and "token_type_ids" in batch:
                kwargs["token_type_ids"] = batch["token_type_ids"].to(device)

            out = model(**kwargs)
            if isinstance(out, dict):
                s = out["start_logits"]
                e = out["end_logits"]
            else:
                s = out.start_logits
                e = out.end_logits
            start_chunks.append(s.detach().cpu().numpy())
            end_chunks.append(e.detach().cpu().numpy())

    start_logits = np.concatenate(start_chunks, axis=0)
    end_logits = np.concatenate(end_chunks, axis=0)
    return start_logits, end_logits


def decode_predictions(
    dataset,
    features,
    start_logits,
    end_logits,
    tokenizer,
    n_best: int,
    max_answer_length: int,
):
    ex_id_to_idx = {k: i for i, k in enumerate(dataset["id"])}
    feat_per_ex = collections.defaultdict(list)
    for i, f in enumerate(features):
        feat_per_ex[ex_id_to_idx[f["example_id"]]].append(i)

    span_text_by_id = {}
    score_diff_by_id = {}

    cls_token_id = tokenizer.cls_token_id
    if cls_token_id is None:
        cls_token_id = tokenizer.bos_token_id
    if cls_token_id is None:
        cls_token_id = tokenizer.pad_token_id
    if cls_token_id is None:
        raise ValueError("Tokenizer has no cls/bos/pad token id for null-score computation.")

    for ex_idx, ex in enumerate(dataset):
        context = ex["context"]
        best_span_score = -1e30
        best_span_text = ""
        best_null_score = 1e30

        for fi in feat_per_ex[ex_idx]:
            sl = start_logits[fi]
            el = end_logits[fi]
            offs = features[fi]["offset_mapping"]
            input_ids = features[fi]["input_ids"]

            if cls_token_id in input_ids:
                cls_idx = input_ids.index(cls_token_id)
            else:
                cls_idx = 0
            null_score = float(sl[cls_idx] + el[cls_idx])
            if null_score < best_null_score:
                best_null_score = null_score

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
                    score = float(sl[s] + el[e])
                    if score > best_span_score:
                        best_span_score = score
                        best_span_text = context[st:en]

        span_text_by_id[ex["id"]] = best_span_text
        score_diff_by_id[ex["id"]] = float(best_null_score - best_span_score)

    return span_text_by_id, score_diff_by_id


def build_predictions_for_threshold(dataset, span_text_by_id, score_diff_by_id, threshold: float):
    preds = []
    for ex in dataset:
        eid = ex["id"]
        text = "" if score_diff_by_id[eid] > threshold else span_text_by_id[eid]
        preds.append({"id": eid, "prediction_text": text, "no_answer_probability": 0.0})
    return preds


def build_predictions_for_prob(dataset, span_text_by_id, score_diff_by_id):
    preds = []
    for ex in dataset:
        eid = ex["id"]
        p_noans = sigmoid(score_diff_by_id[eid])
        preds.append({"id": eid, "prediction_text": span_text_by_id[eid], "no_answer_probability": p_noans})
    return preds


def evaluate_predictions(metric, predictions, references):
    res = metric.compute(predictions=predictions, references=references)
    return {k: float(v) for k, v in res.items()}


def run_single_model(
    spec: ModelRunSpec,
    dataset,
    references,
    metric,
    args,
    device: str,
):
    print(f"[run] {spec.name} ({spec.model_ref})")

    if spec.source == "custom":
        model, tokenizer = load_custom_model_and_tokenizer(
            model_dir=Path(spec.model_ref),
            config_dir=Path(spec.config_ref),
            device=device,
        )
    else:
        model, tokenizer = load_hf_model_and_tokenizer(spec.model_ref, device=device)

    features = tokenize_with_overflow(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        doc_stride=args.doc_stride,
    )
    start_logits, end_logits = collect_logits(
        model=model,
        tokenizer=tokenizer,
        features=features,
        batch_size=args.batch_size,
        device=device,
    )

    span_text_by_id, score_diff_by_id = decode_predictions(
        dataset=dataset,
        features=features,
        start_logits=start_logits,
        end_logits=end_logits,
        tokenizer=tokenizer,
        n_best=args.n_best,
        max_answer_length=args.max_answer_length,
    )

    preds_t0 = build_predictions_for_threshold(dataset, span_text_by_id, score_diff_by_id, threshold=0.0)
    metrics_t0 = evaluate_predictions(metric, preds_t0, references)

    preds_prob = build_predictions_for_prob(dataset, span_text_by_id, score_diff_by_id)
    metrics_prob = evaluate_predictions(metric, preds_prob, references)

    best_exact = metrics_prob.get("best_exact", metrics_prob.get("exact", 0.0))
    best_f1 = metrics_prob.get("best_f1", metrics_prob.get("f1", 0.0))
    best_exact_thresh = metrics_prob.get("best_exact_thresh", None)
    best_f1_thresh = metrics_prob.get("best_f1_thresh", None)

    return {
        "name": spec.name,
        "source": spec.source,
        "model_ref": spec.model_ref,
        "tokenizer_ref": spec.tokenizer_ref,
        "num_examples": len(dataset),
        "metrics_threshold0": metrics_t0,
        "metrics_prob_sweep": metrics_prob,
        "summary": {
            "exact_threshold0": metrics_t0.get("exact", None),
            "f1_threshold0": metrics_t0.get("f1", None),
            "hasans_f1_threshold0": metrics_t0.get("HasAns_f1", None),
            "noans_f1_threshold0": metrics_t0.get("NoAns_f1", None),
            "best_exact": best_exact,
            "best_f1": best_f1,
            "best_exact_thresh": best_exact_thresh,
            "best_f1_thresh": best_f1_thresh,
        },
    }


def write_csv(path: Path, rows):
    cols = [
        "name",
        "source",
        "model_ref",
        "best_f1",
        "best_exact",
        "f1_threshold0",
        "exact_threshold0",
        "hasans_f1_threshold0",
        "noans_f1_threshold0",
        "best_f1_thresh",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            s = r["summary"]
            w.writerow(
                {
                    "name": r["name"],
                    "source": r["source"],
                    "model_ref": r["model_ref"],
                    "best_f1": s.get("best_f1"),
                    "best_exact": s.get("best_exact"),
                    "f1_threshold0": s.get("f1_threshold0"),
                    "exact_threshold0": s.get("exact_threshold0"),
                    "hasans_f1_threshold0": s.get("hasans_f1_threshold0"),
                    "noans_f1_threshold0": s.get("noans_f1_threshold0"),
                    "best_f1_thresh": s.get("best_f1_thresh"),
                }
            )


def write_review(path: Path, rows):
    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]
    if not ok_rows:
        path.write_text("No successful model comparisons were produced.\n", encoding="utf-8")
        return

    ranked = sorted(ok_rows, key=lambda x: x["summary"]["best_f1"], reverse=True)
    best = ranked[0]
    lines = []
    lines.append("SQuAD v2 Comparison Review")
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(f"Top model by best_f1: {best['name']} ({best['summary']['best_f1']:.2f})")
    lines.append("")
    lines.append("Per-model highlights:")
    for r in ranked:
        s = r["summary"]
        lines.append(
            (
                f"- {r['name']}: best_f1={s['best_f1']:.2f}, best_exact={s['best_exact']:.2f}, "
                f"f1@thr0={s['f1_threshold0']:.2f}, hasAns_f1@thr0={s['hasans_f1_threshold0']:.2f}, "
                f"noAns_f1@thr0={s['noans_f1_threshold0']:.2f}"
            )
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "- best_f1 / best_exact are threshold-optimized SQuAD-v2 metrics and are most reliable for model comparison."
    )
    lines.append(
        "- f1@thr0 shows default no-answer behavior; large gaps between f1@thr0 and best_f1 indicate threshold calibration sensitivity."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    specs = []
    if not args.skip_custom:
        specs.append(
            ModelRunSpec(
                name="custom_scratch_encoder_squadv2",
                source="custom",
                model_ref=args.custom_model_dir,
                tokenizer_ref=args.custom_model_dir,
                config_ref=args.custom_config_dir,
            )
        )
    for model_id in [m.strip() for m in args.hf_models.split(",") if m.strip()]:
        safe_name = model_id.replace("/", "_")
        specs.append(
            ModelRunSpec(
                name=f"hf_{safe_name}",
                source="hf",
                model_ref=model_id,
                tokenizer_ref=model_id,
            )
        )

    dataset = build_eval_dataset(args.max_eval_examples)
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in dataset]
    metric = evaluate.load("squad_v2")

    results = {
        "dataset": "squad_v2_validation",
        "num_examples": len(dataset),
        "device": device,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "models": [],
    }

    for spec in specs:
        try:
            model_result = run_single_model(spec, dataset, references, metric, args, device)
            model_result["status"] = "ok"
        except Exception as e:
            model_result = {
                "name": spec.name,
                "source": spec.source,
                "model_ref": spec.model_ref,
                "tokenizer_ref": spec.tokenizer_ref,
                "status": "error",
                "error": str(e),
            }
            print(f"[error] {spec.name}: {e}")
        results["models"].append(model_result)

    ok_rows = [r for r in results["models"] if r.get("status") == "ok"]
    ranked = sorted(ok_rows, key=lambda x: x["summary"]["best_f1"], reverse=True)
    results["ranking_by_best_f1"] = [
        {
            "rank": i + 1,
            "name": r["name"],
            "best_f1": r["summary"]["best_f1"],
            "best_exact": r["summary"]["best_exact"],
            "f1_threshold0": r["summary"]["f1_threshold0"],
        }
        for i, r in enumerate(ranked)
    ]

    out_json = Path(args.output_json)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved JSON: {out_json}")

    out_csv = Path(args.output_csv)
    write_csv(out_csv, ok_rows)
    print(f"Saved CSV: {out_csv}")

    review_txt = Path(args.review_txt)
    write_review(review_txt, results["models"])
    print(f"Saved review: {review_txt}")

    if ranked:
        top = ranked[0]
        print(
            f"Top model: {top['name']} | best_f1={top['summary']['best_f1']:.2f} "
            f"| best_exact={top['summary']['best_exact']:.2f}"
        )
    else:
        print("No successful model runs.")


if __name__ == "__main__":
    main()
