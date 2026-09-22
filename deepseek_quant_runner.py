import argparse
import gc
import json
import os
import re
import subprocess
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import sympy as sp
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    p.add_argument("--kind", choices=["base", "gguf"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-model-len", type=int, default=12288)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def gpu_memory_used_mb():
    """Total GPU memory reported by the driver. Engine telemetry only."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()
        return float(out[0]) if out else np.nan
    except Exception:
        return np.nan


def process_tree_rss_mb():
    """RSS of this runner plus all current child processes."""
    try:
        proc = psutil.Process(os.getpid())
        procs = [proc] + proc.children(recursive=True)
        return sum(p.memory_info().rss for p in procs if p.is_running()) / (1024 ** 2)
    except Exception:
        return np.nan


def system_ram_used_mb():
    return psutil.virtual_memory().used / (1024 ** 2)


class ResourceMonitor:
    def __init__(self, interval=0.25):
        self.interval = interval
        self.stop_event = threading.Event()
        self.gpu = []
        self.process_ram = []
        self.system_ram = []
        self.thread = None

    def _loop(self):
        while not self.stop_event.is_set():
            self.gpu.append(gpu_memory_used_mb())
            self.process_ram.append(process_tree_rss_mb())
            self.system_ram.append(system_ram_used_mb())
            time.sleep(self.interval)

    def start(self):
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)
        return {
            "peak_gpu_engine_mb": float(np.nanmax(self.gpu)) if self.gpu else np.nan,
            "mean_gpu_engine_mb": float(np.nanmean(self.gpu)) if self.gpu else np.nan,
            "peak_process_tree_ram_mb": float(np.nanmax(self.process_ram)) if self.process_ram else np.nan,
            "mean_process_tree_ram_mb": float(np.nanmean(self.process_ram)) if self.process_ram else np.nan,
            "peak_system_ram_mb": float(np.nanmax(self.system_ram)) if self.system_ram else np.nan,
            "mean_system_ram_mb": float(np.nanmean(self.system_ram)) if self.system_ram else np.nan,
        }


def model_artifact_size_bytes(model_spec, kind):
    """
    Storage size of the model artifacts on Hugging Face.
    This is NOT runtime VRAM, but it is a clean precision/quantization size comparison.
    """
    try:
        api = HfApi()
        if kind == "gguf":
            repo_id, quant = model_spec.rsplit(":", 1)
            info = api.model_info(repo_id, files_metadata=True)
            matches = [
                s for s in info.siblings
                if s.rfilename.lower().endswith(".gguf")
                and quant.lower() in s.rfilename.lower()
                and getattr(s, "size", None) is not None
            ]
            if not matches:
                return np.nan, []
            # Prefer the smallest exact quant-matching GGUF if multiple naming variants exist.
            matches = sorted(matches, key=lambda s: s.size)
            chosen = matches[0]
            return int(chosen.size), [chosen.rfilename]
        else:
            info = api.model_info(model_spec, files_metadata=True)
            files = [
                s for s in info.siblings
                if s.rfilename.lower().endswith(".safetensors")
                and getattr(s, "size", None) is not None
            ]
            # Exclude adapter-only files if present.
            files = [s for s in files if "adapter" not in s.rfilename.lower()]
            return int(sum(s.size for s in files)), [s.rfilename for s in files]
    except Exception as e:
        print("WARNING: could not determine model artifact size:", repr(e))
        return np.nan, []


def normalize_text(x):
    if x is None:
        return ""
    s = unicodedata.normalize("NFKD", str(x))
    s = s.lower().strip()
    s = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"\1", s)
    s = s.replace("$", "")
    s = re.sub(r"[^\w\s./+\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_bool_value(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return None
    s = normalize_text(x)
    if s in {"yes", "true", "1", "correct"}:
        return True
    if s in {"no", "false", "0", "incorrect"}:
        return False
    return None


def canonical_math(s):
    s = str(s).strip()
    s = re.sub(r"\\boxed\s*\{(.+?)\}", r"\1", s)
    s = s.replace("$", "").strip()
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    s = s.replace(r"\pi", "pi")
    s = s.replace("^", "**")

    # Convert simple \frac{a}{b} iteratively.
    frac_pat = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s
        s = frac_pat.sub(r"(\1)/(\2)", s)

    # Convert simple \sqrt{a}.
    sqrt_pat = re.compile(r"\\sqrt\s*\{([^{}]+)\}")
    prev = None
    while prev != s:
        prev = s
        s = sqrt_pat.sub(r"sqrt(\1)", s)

    # Preserve tuple/list commas. Only remove thousands separators between digits.
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    return s.strip()


def math_equal(pred, gold):
    p = canonical_math(pred)
    g = canonical_math(gold)

    if normalize_text(p) == normalize_text(g):
        return True

    # Avoid feeding arbitrary long/model-authored code-like strings to SymPy.
    allowed = re.compile(r"^[0-9a-zA-Z_\s+\-*/().,\[\]{}=]*$")
    if len(p) > 256 or len(g) > 256 or not allowed.fullmatch(p) or not allowed.fullmatch(g):
        return False

    # Handle a simple "x = expr" final-answer style by comparing RHS if both have '='.
    if p.count("=") == 1 and g.count("=") == 1:
        p = p.split("=", 1)[1].strip()
        g = g.split("=", 1)[1].strip()

    try:
        # Evaluate only after strict character filtering above.
        return bool(sp.simplify(sp.sympify(p) - sp.sympify(g)) == 0)
    except Exception:
        return False


def parse_bool(pred):
    p = normalize_text(pred)
    first = p.split()[0] if p else ""
    if first in {"yes", "true", "correct"}:
        return True
    if first in {"no", "false", "incorrect"}:
        return False
    return None


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def format_choices(choices):
    if not isinstance(choices, list) or not choices:
        return ""
    return "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(choices))


def clean_choice_wrapper(pred):
    s = str(pred or "").strip()
    s = re.sub(r"^\s*(?:final\s*answer|answer|option|choice)\s*[:\-]?\s*", "", s, flags=re.I)
    s = re.sub(r"^\s*\\boxed\s*\{\s*([A-Za-z])\s*\}\s*$", r"\1", s)
    s = re.sub(r"^\s*\*\*\s*([A-Za-z])\s*\*\*\s*$", r"\1", s)
    s = re.sub(r"^\s*[\(\[\{]\s*([A-Za-z])\s*[\)\]\}]\s*$", r"\1", s)
    s = re.sub(r"^\s*([A-Za-z])\s*[\).:\-]\s*", r"\1 ", s)
    return s.strip()


def extract_choice_index(pred, choices):
    if not isinstance(choices, list) or not choices:
        return None

    cleaned = clean_choice_wrapper(pred)

    # Accept B, "B text", "B. text", etc.
    m = re.match(r"^\s*([A-Z])(?:\b|[\s.):,\-])", cleaned, flags=re.I)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return idx

    # Also search for explicit "option B"/"answer B" in a concise final field.
    m = re.search(r"\b(?:option|answer|choice)\s*[:\-]?\s*([A-Z])\b", str(pred), flags=re.I)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return idx

    pred_norm = normalize_text(pred)
    exact = [i for i, choice in enumerate(choices) if pred_norm == normalize_text(choice)]
    if len(exact) == 1:
        return exact[0]

    contained = [
        i for i, choice in enumerate(choices)
        if normalize_text(choice) and normalize_text(choice) in pred_norm
    ]
    if len(contained) == 1:
        return contained[0]
    return None


def gold_choice_index(gold, choices, index_base=None):
    """Normalize dict/letter/int/text MCQ gold formats."""
    n = len(choices) if isinstance(choices, list) else 0

    if isinstance(gold, dict):
        # Prefer explicit letter/text when available.
        for key in ("letter", "label", "answer", "choice"):
            if key in gold:
                v = gold[key]
                if isinstance(v, str):
                    vv = v.strip().upper()
                    if len(vv) == 1 and vv in LETTERS[:n]:
                        return LETTERS.index(vv)
                    if v in choices:
                        return choices.index(v)
        if "index" in gold:
            idx = gold["index"]
            try:
                idx = int(idx)
            except Exception:
                return None
            if index_base == 1 and 1 <= idx <= n:
                return idx - 1
            if index_base == 0 and 0 <= idx < n:
                return idx
            # Fallback only when the value itself disambiguates the base.
            if idx == 0:
                return 0
            if idx == n:
                return n - 1
            return None

    if isinstance(gold, str):
        g = gold.strip()
        gu = g.upper()
        if len(gu) == 1 and gu in LETTERS[:n]:
            return LETTERS.index(gu)
        if g in choices:
            return choices.index(g)
        if re.fullmatch(r"\d+", g):
            idx = int(g)
            if index_base == 1 and 1 <= idx <= n:
                return idx - 1
            if index_base == 0 and 0 <= idx < n:
                return idx
            if idx == 0:
                return 0
            if idx == n:
                return n - 1

    if isinstance(gold, (int, np.integer)):
        idx = int(gold)
        if index_base == 1 and 1 <= idx <= n:
            return idx - 1
        if index_base == 0 and 0 <= idx < n:
            return idx
        if idx == 0:
            return 0
        if idx == n:
            return n - 1

    return None



def infer_index_base(rows, dataset):
    """
    Infer index base from a whole dataset when numeric/dict indices are used.
    Presence of 0 proves 0-based; presence of N proves 1-based.
    If still ambiguous, default to 0-based but emit a visible warning.
    """
    observed = []
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        gold = row.get("answer")
        choices = row.get("choices")
        n = len(choices) if isinstance(choices, list) else 0
        value = None
        if isinstance(gold, dict) and "index" in gold:
            value = gold.get("index")
        elif isinstance(gold, (int, np.integer)):
            value = gold
        elif isinstance(gold, str) and re.fullmatch(r"\d+", gold.strip()):
            value = gold.strip()
        if value is not None:
            try:
                observed.append((int(value), n))
            except Exception:
                pass

    if any(v == 0 for v, _ in observed):
        return 0
    if any(n > 0 and v == n for v, n in observed):
        return 1
    if observed:
        print(f"WARNING: {dataset} numeric gold index base is ambiguous; defaulting to 0-based.")
        return 0
    return None


def build_user_instruction(row):
    dataset = row["dataset"]
    rules = {
        "math500": (
            "Solve the problem carefully. End with `FINAL_ANSWER:` followed by only "
            "the final value or expression. A boxed answer is also acceptable."
        ),
        "strategyqa": (
            "Reason carefully. End with `FINAL_ANSWER: Yes` or `FINAL_ANSWER: No`."
        ),
        "logiqa2": (
            "Choose the best option. End with `FINAL_ANSWER:` followed by the option "
            "letter and optionally the option text."
        ),
        "date_understanding": (
            "Determine the correct date. End with `FINAL_ANSWER:` followed by the "
            "option letter and/or the exact date."
        ),
        "simpleqa_verified": (
            "Answer the factual question precisely. End with `FINAL_ANSWER:` followed "
            "by a concise answer only."
        ),
        "misguided_attention": (
            "Inspect the premise carefully before solving. Do not assume the obvious "
            "interpretation is valid. End with `FINAL_ANSWER:` followed by your concise conclusion."
        ),
    }

    prompt = (
        "Please reason step by step before answering.\n\n"
        f"Task instruction: {rules.get(dataset, 'Answer carefully and give a concise final answer.')}\n\n"
        f"Question:\n{row['question']}\n"
    )

    choice_text = format_choices(row.get("choices"))
    if choice_text:
        prompt += f"\nOptions:\n{choice_text}\n"

    prompt += "\nDo not omit the `FINAL_ANSWER:` line."
    return prompt


FINAL_RE = re.compile(r"FINAL_ANSWER\s*:\s*([^\r\n]+)", flags=re.I)


def ensure_reasoning_start(rendered):
    """Avoid duplicating <think> if the tokenizer template already emitted it."""
    tail = rendered.rstrip()
    if re.search(r"<think>\s*$", tail, flags=re.I):
        return rendered
    return rendered + "<think>\n"


def split_reasoning_and_answer(text):
    text = text or ""

    if "</think>" in text:
        reasoning, post = text.split("</think>", 1)
    else:
        m = FINAL_RE.search(text)
        if m:
            reasoning = text[:m.start()]
            post = text[m.start():]
        else:
            reasoning = text
            post = ""

    reasoning = reasoning.replace("<think>", "").strip()

    fm = FINAL_RE.search(post if post else text)
    if fm:
        final = fm.group(1).strip()
        parse_status = "final_answer_found"
    else:
        # Crucial: never reinterpret reasoning text as the final answer.
        final = ""
        parse_status = "missing_final_answer"

    return reasoning, final, parse_status


def evaluate_row(row, final_answer, parse_success):
    dataset = row["dataset"]
    gold = row.get("answer")
    choices = row.get("choices")

    # End-to-end scoring: failure to produce a parseable final answer is incorrect
    # for objectively scored datasets, but is tracked separately from reasoning error.
    if not parse_success:
        if dataset == "misguided_attention":
            return np.nan, "rubric_requires_review"
        return False, "missing_final_answer"

    if dataset == "strategyqa":
        pred_bool = parse_bool(final_answer)
        gold_bool = normalize_bool_value(gold)
        if gold_bool is None:
            return np.nan, "invalid_gold_boolean"
        return ((pred_bool == gold_bool) if pred_bool is not None else False), "boolean"

    if dataset == "logiqa2":
        pred_idx = extract_choice_index(final_answer, choices)
        gold_idx = gold_choice_index(gold, choices, row.get("_index_base"))
        if gold_idx is None:
            return np.nan, "invalid_gold_choice"
        return pred_idx == gold_idx, "multiple_choice"

    if dataset == "date_understanding":
        pred_idx = extract_choice_index(final_answer, choices)
        gold_idx = gold_choice_index(gold, choices, row.get("_index_base"))
        if pred_idx is not None and gold_idx is not None:
            return pred_idx == gold_idx, "multiple_choice_date"
        return normalize_text(final_answer) == normalize_text(gold), "date_exact"

    if dataset == "math500":
        return math_equal(final_answer, gold), "symbolic_math"

    if dataset == "simpleqa_verified":
        p = normalize_text(final_answer)
        g = normalize_text(gold)
        # FINAL_ANSWER is already concise; allow exact match or a single unambiguous
        # gold span in the final field, while avoiding reasoning-text contamination.
        return (p == g) or (bool(g) and re.search(rf"\b{re.escape(g)}\b", p) is not None), "normalized_fact"

    if dataset == "misguided_attention":
        return np.nan, "rubric_requires_review"

    return np.nan, "unsupported"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_existing_records(path):
    if not path.exists():
        return []
    records = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        print("WARNING: could not restore checkpoint:", repr(e))
        return []
    return records


def write_checkpoint(records, jsonl_path, csv_path):
    tmp_jsonl = jsonl_path.with_suffix(".tmp.jsonl")
    with open(tmp_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp_jsonl, jsonl_path)

    tmp_csv = csv_path.with_suffix(".tmp.csv")
    pd.DataFrame(records).to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, csv_path)


def main():
    args = parse_args()

    out_dir = Path(args.output_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = out_dir / "answers_and_evaluations.jsonl"
    result_csv = out_dir / "answers_and_evaluations.csv"
    resource_json = out_dir / "resource_summary.json"
    status_json = out_dir / "status.json"
    prompt_preview = out_dir / "rendered_prompt_preview.txt"

    rows = load_jsonl(args.input)

    for dataset_name in ("logiqa2", "date_understanding"):
        inferred_base = infer_index_base(rows, dataset_name)
        if inferred_base is not None:
            print(f"{dataset_name} inferred numeric gold index base: {inferred_base}")
        for row in rows:
            if row.get("dataset") == dataset_name:
                row["_index_base"] = inferred_base

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for row in rows:
        user_prompt = build_user_instruction(row)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        row["_prompt"] = ensure_reasoning_start(rendered)

    # Save one rendered prompt so the <think> behavior is auditable.
    if rows:
        prompt_preview.write_text(rows[0]["_prompt"], encoding="utf-8")
        print("Prompt tail:", repr(rows[0]["_prompt"][-120:]))

    existing_records = load_existing_records(result_jsonl)
    valid_ids = {str(r["sample_id"]) for r in rows}
    existing_records = [
        r for r in existing_records
        if str(r.get("sample_id")) in valid_ids and r.get("precision") == args.label
    ]
    completed_ids = {str(r["sample_id"]) for r in existing_records}
    remaining_rows = [r for r in rows if str(r["sample_id"]) not in completed_ids]

    print(f"[{args.label}] restored {len(existing_records)} completed questions; "
          f"{len(remaining_rows)} remain.")

    status = {
        "label": args.label,
        "kind": args.kind,
        "model": args.model,
        "dtype": args.dtype,
        "state": "loading_model",
        "n_questions": len(rows),
        "restored_questions": len(existing_records),
        "started_unix": time.time(),
    }
    status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    artifact_bytes, artifact_files = model_artifact_size_bytes(args.model, args.kind)

    gpu_before = gpu_memory_used_mb()
    process_ram_before = process_tree_rss_mb()
    system_ram_before = system_ram_used_mb()

    llm_kwargs = dict(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        seed=args.seed,
        tensor_parallel_size=1,
    )

    # Current GGUF plugin supports repo_id:quant_type. hf_config_path is useful
    # for GGUF repos whose metadata cannot be mapped reliably to HF config.
    if args.kind == "gguf":
        llm_kwargs["hf_config_path"] = args.tokenizer

    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    gpu_after_load = gpu_memory_used_mb()
    process_ram_after_load = process_tree_rss_mb()

    monitor = ResourceMonitor(interval=0.25).start()
    records = list(existing_records)
    session_output_tokens = 0
    run_start = time.perf_counter()

    status["state"] = "running"
    status["completed_questions"] = len(records)
    status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    # Continue batch numbering after any restored records.
    next_batch_index = 0
    if records:
        prior_batches = [
            int(r.get("batch_index", -1)) for r in records
            if str(r.get("batch_index", "")).lstrip("-").isdigit()
        ]
        if prior_batches:
            next_batch_index = max(prior_batches) + 1

    for offset in range(0, len(remaining_rows), args.batch_size):
        chunk = remaining_rows[offset:offset + args.batch_size]
        prompts = [r["_prompt"] for r in chunk]

        batch_index = next_batch_index + (offset // args.batch_size)
        chunk_start = time.perf_counter()
        outputs = llm.generate(prompts, sampling, use_tqdm=False)
        chunk_elapsed = time.perf_counter() - chunk_start

        for row, output in zip(chunk, outputs):
            completion = output.outputs[0]
            generated = completion.text or ""
            finish_reason = getattr(completion, "finish_reason", None)
            stop_reason = getattr(completion, "stop_reason", None)
            was_truncated = str(finish_reason).lower() == "length"

            reasoning, final_answer, parse_status = split_reasoning_and_answer(generated)
            parse_success = parse_status == "final_answer_found"

            prompt_tokens = (
                len(output.prompt_token_ids)
                if output.prompt_token_ids is not None
                else len(tokenizer.encode(row["_prompt"], add_special_tokens=False))
            )
            output_tokens = (
                len(completion.token_ids)
                if completion.token_ids is not None
                else len(tokenizer.encode(generated, add_special_tokens=False))
            )
            reasoning_tokens = len(tokenizer.encode(reasoning, add_special_tokens=False))
            answer_tokens = len(tokenizer.encode(final_answer, add_special_tokens=False)) if final_answer else 0

            correct, eval_method = evaluate_row(row, final_answer, parse_success)
            session_output_tokens += output_tokens

            records.append({
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "category": row["category"],
                "difficulty": row["difficulty"],
                "source_split": row.get("source_split"),
                "source_index": row.get("source_index"),
                "question": row["question"],
                "choices": row.get("choices"),
                "gold_answer": row.get("answer"),
                "evaluation_criteria": row.get("evaluation_criteria"),

                "precision": args.label,
                "model_kind": args.kind,
                "model_spec": args.model,
                "dtype_argument": args.dtype,

                "generated_text": generated,
                "reasoning_text": reasoning,
                "final_answer": final_answer,

                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "was_truncated": was_truncated,
                "parse_status": parse_status,
                "parse_success": parse_success,

                "correct": correct,
                "evaluation_method": eval_method,

                "prompt_tokens": prompt_tokens,
                "reasoning_tokens": reasoning_tokens,
                "reasoning_words": len(re.findall(r"\S+", reasoning)),
                "reasoning_chars": len(reasoning),
                "answer_tokens": answer_tokens,
                "output_tokens": output_tokens,
                "total_tokens": prompt_tokens + output_tokens,

                # Batch timing is telemetry only. Headline speed uses whole-run throughput.
                "batch_index": batch_index,
                "batch_wall_time_s": chunk_elapsed,
                "batch_size_actual": len(outputs),

                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            })

        write_checkpoint(records, result_jsonl, result_csv)

        status["completed_questions"] = len(records)
        status["state"] = "running"
        status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

        print(
            f"[{args.label}] {len(records)}/{len(rows)} saved to Drive "
            f"| last batch {chunk_elapsed:.2f}s"
        )

    run_elapsed = time.perf_counter() - run_start
    resources = monitor.stop()

    # Re-read all records for totals, including restored checkpoint records.
    total_output_tokens_all = int(sum(int(r.get("output_tokens", 0) or 0) for r in records))
    total_truncated = int(sum(bool(r.get("was_truncated", False)) for r in records))
    total_parse_failures = int(sum(not bool(r.get("parse_success", False)) for r in records))

    resource_summary = {
        "precision": args.label,
        "kind": args.kind,
        "model": args.model,
        "dtype_argument": args.dtype,
        "n_questions": len(records),
        "restored_questions": len(existing_records),
        "session_questions": len(remaining_rows),
        "session_run_wall_time_s": run_elapsed,
        "session_questions_per_second": len(remaining_rows) / max(run_elapsed, 1e-9),
        "session_output_tokens": session_output_tokens,
        "session_output_tokens_per_second": session_output_tokens / max(run_elapsed, 1e-9),
        "total_output_tokens_all_records": total_output_tokens_all,
        "truncated_count": total_truncated,
        "parse_failure_count": total_parse_failures,

        "model_artifact_size_bytes": None if np.isnan(artifact_bytes) else int(artifact_bytes),
        "model_artifact_size_gb": None if np.isnan(artifact_bytes) else float(artifact_bytes / (1024 ** 3)),
        "model_artifact_files": artifact_files,

        "gpu_before_load_mb": gpu_before,
        "gpu_after_load_mb": gpu_after_load,
        "gpu_engine_load_delta_mb": (
            gpu_after_load - gpu_before
            if not (np.isnan(gpu_after_load) or np.isnan(gpu_before))
            else np.nan
        ),
        "process_tree_ram_before_load_mb": process_ram_before,
        "process_tree_ram_after_load_mb": process_ram_after_load,
        "system_ram_before_load_mb": system_ram_before,

        "gpu_memory_note": (
            "nvidia-smi values include vLLM engine/KV-cache reservation and must not be "
            "interpreted as pure model-weight VRAM."
        ),
        **resources,
    }
    resource_json.write_text(json.dumps(resource_summary, indent=2), encoding="utf-8")

    status["state"] = "complete"
    status["completed_questions"] = len(records)
    status["finished_unix"] = time.time()
    status_json.write_text(json.dumps(status, indent=2), encoding="utf-8")

    print(json.dumps(resource_summary, indent=2))

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
