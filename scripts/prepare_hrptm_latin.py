#!/usr/bin/env python
"""Prepare Latin-script HRPTM corpus JSONL for ELF conditional training (t5-small).

Reads merged train/eval JSONL (`input` / `output` fields) and writes Hugging Face
Arrow datasets with `condition_input_ids`, `input_ids`, `input`, `target` matching
`prepare_armman_hindi.py` / `get_dataloader` expectations.

Example:
  python scripts/prepare_hrptm_latin.py \\
    --train_jsonl /path/to/merged_latin_train.jsonl \\
    --eval_jsonl /path/to/merged_latin_eval.jsonl \\
    --out_dir data/hrptm_t5_small_latin
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from datasets import Dataset
from transformers import T5Tokenizer

DEFAULT_SYSTEM = (
    "You are a helpful assistant for ANM (Auxiliary Nurse Midwife) workers. "
    "Answer health and medical queries accurately and concisely."
)


def _format_prompt(system: str, user: str) -> str:
    return f"{system}\n\nUser:\n{user}"


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            inp = obj.get("input")
            out = obj.get("output")
            if not isinstance(inp, str) or not isinstance(out, str):
                continue
            inp, out = inp.strip(), out.strip()
            if not inp or not out:
                continue
            rows.append({"line_no": line_no, "input": inp, "output": out})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare HRPTM Latin ELF Arrow dataset from JSONL.")
    here = os.path.dirname(os.path.dirname(__file__))
    parser.add_argument(
        "--train_jsonl",
        type=str,
        default=os.path.join(
            here,
            "..",
            "prod-HRPTM-main",
            "data",
            "processed",
            "merged_latin_train.jsonl",
        ),
    )
    parser.add_argument(
        "--eval_jsonl",
        type=str,
        default=os.path.join(
            here,
            "..",
            "prod-HRPTM-main",
            "data",
            "processed",
            "merged_latin_eval.jsonl",
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(here, "data", "hrptm_t5_small_latin"),
    )
    parser.add_argument("--tokenizer_name", type=str, default="t5-small")
    parser.add_argument("--max_input_length", type=int, default=192)
    parser.add_argument("--max_target_length", type=int, default=320)
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM)
    args = parser.parse_args()

    train_jsonl = os.path.abspath(args.train_jsonl)
    eval_jsonl = os.path.abspath(args.eval_jsonl)

    tok = T5Tokenizer.from_pretrained(args.tokenizer_name)
    train_raw = _load_jsonl(train_jsonl)
    eval_raw = _load_jsonl(eval_jsonl)
    if not train_raw:
        print(f"No rows in {train_jsonl}", file=sys.stderr)
        sys.exit(1)

    def build_split(rows: List[Dict[str, Any]], split_name: str):
        out = []
        audit = []
        for i, ex in enumerate(rows):
            prompt = _format_prompt(args.system_prompt, ex["input"])
            cond_ids = tok(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_input_length,
            )["input_ids"]
            tgt_ids = tok(
                ex["output"],
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_target_length,
            )["input_ids"]
            out.append(
                {
                    "condition_input_ids": cond_ids,
                    "input_ids": tgt_ids,
                    "input": prompt,
                    "target": ex["output"],
                }
            )
            audit.append(
                {
                    "split": split_name,
                    "index": i,
                    "source_line_no": ex["line_no"],
                    "len_condition_tokens": len(cond_ids),
                    "len_answer_tokens": len(tgt_ids),
                    "len_total_tokens": len(cond_ids) + len(tgt_ids),
                }
            )
        return out, audit

    train_tok, train_audit = build_split(train_raw, "train")
    eval_tok, eval_audit = build_split(eval_raw, "eval")

    train_ds = Dataset.from_list(train_tok)
    eval_ds = Dataset.from_list(eval_tok)

    os.makedirs(args.out_dir, exist_ok=True)
    train_out = os.path.join(args.out_dir, "train")
    eval_out = os.path.join(args.out_dir, "eval")
    train_ds.save_to_disk(train_out)
    eval_ds.save_to_disk(eval_out)

    audit_path = os.path.join(args.out_dir, "audit.jsonl")
    with open(audit_path, "w", encoding="utf-8") as af:
        for row in train_audit + eval_audit:
            af.write(json.dumps(row) + "\n")

    lens_tr = [r["len_total_tokens"] for r in train_audit]
    print(f"Train examples: {len(train_ds)} -> {train_out}")
    print(f"Eval examples:  {len(eval_ds)} -> {eval_out}")
    print(f"Audit -> {audit_path}")
    if lens_tr:
        print(
            f"Train token totals (cond+ans): min={min(lens_tr)} max={max(lens_tr)} "
            f"mean={sum(lens_tr)/len(lens_tr):.1f}"
        )


if __name__ == "__main__":
    main()
