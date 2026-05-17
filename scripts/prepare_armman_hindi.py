#!/usr/bin/env python
"""Build a small Hindi/Devanagari ARMMAN slice for ELF conditional training.

Reads chat JSONL (`text` + `messages`) and keeps rows whose assistant answer
contains Devanagari script. Writes Hugging Face Arrow datasets compatible with
`utils.data_utils.get_dataloader`: `condition_input_ids` + `input_ids`.

Example:
  python scripts/prepare_armman_hindi.py \\
    --src data/armman/train_qwen3.jsonl \\
    --out_dir data/armman_t5_small_hindi \\
    --n_train 128 --n_eval 16 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset
from transformers import T5Tokenizer


def _has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def _extract_roles(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    system = user = assistant = None
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = m.get("content")
        if not isinstance(content, str):
            continue
        if role == "system":
            system = content.strip()
        elif role == "user":
            user = content.strip()
        elif role == "assistant":
            assistant = content.strip()
    return system, user, assistant


def _format_prompt(system: str, user: str) -> str:
    return f"{system}\n\nUser:\n{user}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ARMMAN Hindi ELF Arrow dataset from JSONL.")
    parser.add_argument(
        "--src",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "armman", "train_qwen3.jsonl"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "armman_t5_small_hindi"),
    )
    parser.add_argument("--n_train", type=int, default=128)
    parser.add_argument("--n_eval", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer_name", type=str, default="t5-small")
    parser.add_argument("--max_input_length", type=int, default=192, help="Max tokens for condition (prompt).")
    parser.add_argument("--max_target_length", type=int, default=320, help="Max tokens for assistant answer.")
    args = parser.parse_args()

    tok = T5Tokenizer.from_pretrained(args.tokenizer_name)

    candidates: List[Dict[str, Any]] = []
    with open(args.src, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            messages = row.get("messages")
            if not isinstance(messages, list):
                continue
            system, user, assistant = _extract_roles(messages)
            if not system or not user or not assistant:
                continue
            if not _has_devanagari(assistant):
                continue
            prompt = _format_prompt(system, user)
            candidates.append(
                {
                    "line_no": line_no,
                    "input": prompt,
                    "output": assistant,
                    "user_devanagari": _has_devanagari(user),
                }
            )

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    need = args.n_train + args.n_eval
    if len(candidates) < need:
        raise SystemExit(
            f"Only {len(candidates)} Hindi-assistant examples found; need at least {need}. "
            f"Check --src or relax the Devanagari filter."
        )

    train_raw = candidates[: args.n_train]
    eval_raw = candidates[args.n_train : args.n_train + args.n_eval]

    def build_split(rows: List[Dict[str, Any]], split_name: str) -> Tuple[List[Dict], List[Dict]]:
        out: List[Dict] = []
        audit: List[Dict] = []
        for i, ex in enumerate(rows):
            cond_ids = tok(
                ex["input"],
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
            # String fields required by `generation.test_generation_cond` (log reference vs generated).
            out.append(
                {
                    "condition_input_ids": cond_ids,
                    "input_ids": tgt_ids,
                    "input": ex["input"],
                    "target": ex["output"],
                }
            )
            audit.append(
                {
                    "split": split_name,
                    "index": i,
                    "source_line_no": ex["line_no"],
                    "input": ex["input"],
                    "output": ex["output"],
                    "user_has_devanagari": ex["user_devanagari"],
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

    train_out = os.path.join(args.out_dir, "train")
    eval_out = os.path.join(args.out_dir, "eval")
    os.makedirs(args.out_dir, exist_ok=True)
    train_ds.save_to_disk(train_out)
    eval_ds.save_to_disk(eval_out)

    audit_path = os.path.join(args.out_dir, "audit.jsonl")
    with open(audit_path, "w", encoding="utf-8") as af:
        for row in train_audit + eval_audit:
            af.write(json.dumps(row, ensure_ascii=False) + "\n")

    lens_train = [r["len_total_tokens"] for r in train_audit]
    print(f"Candidates (Hindi assistant): {len(candidates)}")
    print(f"Saved train ({len(train_ds)} ex) -> {train_out}")
    print(f"Saved eval  ({len(eval_ds)} ex) -> {eval_out}")
    print(f"Audit       -> {audit_path}")
    print(
        f"Token length stats (train concat cond+ans): "
        f"min={min(lens_train)} max={max(lens_train)} "
        f"mean={sum(lens_train)/len(lens_train):.1f}"
    )
    print("Example (first train audit row, truncated):")
    ex0 = train_audit[0]
    print("  input[:200]:", ex0["input"][:200].replace("\n", "\\n"))
    print("  output[:200]:", ex0["output"][:200].replace("\n", "\\n"))


if __name__ == "__main__":
    main()
