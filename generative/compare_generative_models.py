import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from generative_data import NO_ANSWER_TEXT, normalize_text
from standard_generative_decoder import DecoderConfig, GenerativeQAModel as StandardGenerativeQAModel
from main_hybrid_decoder import GenerativeQAModelHybrid
from mlm_pretraining import ModelConfig


DEFAULT_HF_MODELS = ["t5-small", "t5-base", "google/flan-t5-small"]


@dataclass
class ModelRunSpec:
    name: str
    kind: str  # "custom" or "hf"
    ref: str
    tokenizer_ref: str
    decoder_variant: str = "hybrid"


def parse_args():
    p = argparse.ArgumentParser(description="Compare generative QA models on SQuAD v2 validation.")
    p.add_argument("--custom_checkpoint", default="checkpoints_generative_qa_hybrid_span_restart1_20260426_010636/latest.pt")
    p.add_argument("--custom_tokenizer", default="checkpoints_pretrain_base_seq256/step_20000")
    p.add_argument("--custom_name", default="our_hybrid_decoder")
    p.add_argument("--hf_models", default=",".join(DEFAULT_HF_MODELS))
    p.add_argument("--max_input_len", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=12)
    p.add_argument("--beam_size", type=int, default=1)
    p.add_argument("--length_penalty", type=float, default=0.4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_eval_examples", type=int, default=4000)
    p.add_argument("--instruction_prefix", default="")
    p.add_argument("--no_answer_text", default=NO_ANSWER_TEXT)
    p.add_argument("--output_json", default="comparison_generative_seq2seq.json")
    p.add_argument("--output_csv", default="comparison_generative_seq2seq.csv")
    return p.parse_args()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_text(pred) == normalize_text(gold))


def f1_score(pred: str, gold: str) -> float:
    p_toks = normalize_text(pred).split()
    g_toks = normalize_text(gold).split()
    common = {}
    for t in p_toks:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in g_toks:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / max(1, len(p_toks))
    rec = overlap / max(1, len(g_toks))
    return 2 * prec * rec / max(1e-8, prec + rec)


def build_prompt(question: str, context: str, instruction_prefix: str = "") -> str:
    if instruction_prefix:
        return f"{instruction_prefix.strip()} question: {question} context: {context}"
    return f"question: {question} context: {context}"


def load_eval_dataset(max_eval_examples: int):
    ds = load_dataset("squad_v2", split="validation")
    if max_eval_examples > 0:
        ds = ds.select(range(min(max_eval_examples, len(ds))))
    return ds


def load_custom_model(checkpoint_path: str, tokenizer_path: str, device: str, decoder_variant: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    enc_cfg = ModelConfig(**payload["encoder_config"])
    dec_cfg = DecoderConfig(**payload["decoder_config"])
    model = GenerativeQAModelHybrid(enc_cfg, dec_cfg) if decoder_variant == "hybrid" else StandardGenerativeQAModel(enc_cfg, dec_cfg)
    model.load_state_dict(payload["model"], strict=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.sep_token
    model.to(device).eval()
    return model, tokenizer


def load_hf_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model.to(device).eval()
    return model, tokenizer


def decode_custom(tokenizer, out_ids, bos: int, eos: int, pad: int) -> str:
    text_ids = []
    for tid in out_ids:
        if tid in {bos, pad}:
            continue
        if tid == eos:
            break
        text_ids.append(tid)
    return tokenizer.decode(text_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def run_model(spec: ModelRunSpec, dataset, args, device: str):
    if spec.kind == "custom":
        model, tokenizer = load_custom_model(spec.ref, spec.tokenizer_ref, device, spec.decoder_variant)
        is_custom = True
    else:
        model, tokenizer = load_hf_model(spec.ref, device)
        is_custom = False

    preds = []
    golds = []
    is_noans_flags = []
    total_len = 0

    bos = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    eos = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    for start in range(0, len(dataset), args.batch_size):
        batch = dataset.select(range(start, min(start + args.batch_size, len(dataset))))
        questions = [ex["question"] for ex in batch]
        contexts = [ex["context"] for ex in batch]
        targets = []
        inputs = [build_prompt(q, c, args.instruction_prefix) for q, c in zip(questions, contexts)]
        for ex in batch:
            ans = ex["answers"]["text"]
            gold = args.no_answer_text if len(ans) == 0 else ans[0].strip()
            targets.append(gold)
            is_noans_flags.append(len(ans) == 0)

        if is_custom:
            batch_preds = []
            for inp in inputs:
                enc = tokenizer(
                    [inp],
                    truncation=True,
                    max_length=args.max_input_len,
                    padding=True,
                    return_tensors="pt",
                )
                enc_ids = enc["input_ids"].to(device)
                enc_mask = enc["attention_mask"].to(device)
                enc_ttype = enc.get("token_type_ids", torch.zeros_like(enc_ids)).to(device)
                out = model.generate(
                    encoder_input_ids=enc_ids,
                    encoder_token_type_ids=enc_ttype,
                    encoder_attention_mask=enc_mask,
                    bos_token_id=bos,
                    eos_token_id=eos,
                    pad_token_id=pad,
                    max_new_tokens=args.max_new_tokens,
                    beam_size=args.beam_size,
                    length_penalty=args.length_penalty,
                )
                batch_preds.append(decode_custom(tokenizer, out[0].tolist(), bos=bos, eos=eos, pad=pad))
        else:
            enc = tokenizer(
                inputs,
                truncation=True,
                max_length=args.max_input_len,
                padding=True,
                return_tensors="pt",
            )
            enc_ids = enc["input_ids"].to(device)
            enc_mask = enc["attention_mask"].to(device)
            out = model.generate(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.beam_size,
                length_penalty=args.length_penalty,
                early_stopping=True,
            )
            batch_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
            batch_preds = [x.strip() for x in batch_preds]

        for pred, gold in zip(batch_preds, targets):
            preds.append(pred)
            golds.append(gold)
            total_len += len(pred.split())

    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    rouge_res = rouge.compute(predictions=preds, references=golds)
    bleu_res = bleu.compute(predictions=preds, references=[[g] for g in golds])

    ems = [exact_match(p, g) for p, g in zip(preds, golds)]
    f1s = [f1_score(p, g) for p, g in zip(preds, golds)]

    noans_total = sum(is_noans_flags)
    ans_total = len(is_noans_flags) - noans_total
    noans_correct = sum(
        1 for p, is_noans in zip(preds, is_noans_flags) if is_noans and normalize_text(p) == normalize_text(args.no_answer_text)
    )
    ans_correct = sum(
        1 for p, is_noans in zip(preds, is_noans_flags) if (not is_noans) and normalize_text(p) != normalize_text(args.no_answer_text)
    )

    return {
        "name": spec.name,
        "kind": spec.kind,
        "model_ref": spec.ref,
        "num_examples": len(dataset),
        "exact_match": 100.0 * sum(ems) / max(1, len(ems)),
        "f1": 100.0 * sum(f1s) / max(1, len(f1s)),
        "rougeL": 100.0 * rouge_res["rougeL"],
        "bleu": 100.0 * bleu_res["bleu"],
        "no_answer_accuracy": 100.0 * noans_correct / max(1, noans_total),
        "answerable_accuracy": 100.0 * ans_correct / max(1, ans_total),
        "avg_output_len": total_len / max(1, len(preds)),
        "sample_predictions": [
            {"question": dataset[i]["question"], "gold": golds[i], "pred": preds[i]}
            for i in range(min(5, len(preds)))
        ],
    }


def write_csv(path: Path, rows):
    cols = [
        "name",
        "model_ref",
        "exact_match",
        "f1",
        "rougeL",
        "bleu",
        "no_answer_accuracy",
        "answerable_accuracy",
        "avg_output_len",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dataset = load_eval_dataset(args.max_eval_examples)
    specs = [
        ModelRunSpec(
            name=args.custom_name,
            kind="custom",
            ref=args.custom_checkpoint,
            tokenizer_ref=args.custom_tokenizer,
            decoder_variant="hybrid",
        )
    ]
    for model_id in [m.strip() for m in args.hf_models.split(",") if m.strip()]:
        specs.append(
            ModelRunSpec(
                name=model_id.replace("/", "_"),
                kind="hf",
                ref=model_id,
                tokenizer_ref=model_id,
            )
        )

    results = {
        "dataset": "squad_v2_validation",
        "num_examples": len(dataset),
        "device": device,
        "args": vars(args),
        "models": [],
    }

    for spec in specs:
        print(f"[run] {spec.name} ({spec.ref})")
        try:
            res = run_model(spec, dataset, args, device)
            res["status"] = "ok"
            print(json.dumps(res, indent=2))
        except Exception as e:
            res = {"name": spec.name, "model_ref": spec.ref, "status": "error", "error": str(e)}
            print(f"[error] {spec.name}: {e}")
        results["models"].append(res)

    ok_rows = [r for r in results["models"] if r.get("status") == "ok"]
    ranked = sorted(ok_rows, key=lambda x: x["f1"], reverse=True)
    results["ranking_by_f1"] = [
        {"rank": i + 1, "name": r["name"], "f1": r["f1"], "exact_match": r["exact_match"]}
        for i, r in enumerate(ranked)
    ]

    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(Path(args.output_csv), ok_rows)
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved CSV: {args.output_csv}")
    if ranked:
        print(f"Top model: {ranked[0]['name']} | F1={ranked[0]['f1']:.2f} | EM={ranked[0]['exact_match']:.2f}")


if __name__ == "__main__":
    main()
