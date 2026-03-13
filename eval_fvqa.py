"""
todo: this script terminates after sometime, probably due to api instability. test for locally deployed models.
Evaluate RLM with image search on the FVQA test set.

Each sample has a query image and a factual question.  Pre-computed web search
results are loaded from a local search_cache.pkl so no live internet search is
needed.  The agent can:
    1. Inspect the query image with view_image().
    2. Call image_search(question_id) to get cached web results.
    3. Fetch and inspect individual thumbnails via result.get_thumbnail(i) +
       view_image().

Data layout::

    /path/to/FVQA/
        fvqa_test.parquet   — rows: prompt, images (bytes), data_id, reward_model
        search_cache.pkl    — dict[data_id -> {titles, image_urls}]

Environment variables::

    FVQA_DIR            Path to the FVQA data directory (required)
    RLM_MODEL_NAME      Model name  (default: gemini-2.5-pro-preview)
    RLM_BASE_URL        OpenAI-compatible base URL (default: https://yunwu.ai/v1)
    RLM_API_KEY_ENV     Name of the env-var holding the API key (default: YUNWU_API_KEY)
    MAX_EXAMPLES        Cap on the number of samples to evaluate
    START_IDX           First sample index (0-based, default: 0)
    END_IDX             One-past-last sample index (default: all)
    OUTPUT_FILE         Path for the JSONL output (default: fvqa_results.jsonl)
    WORKERS             Number of concurrent API workers (default: 8)

Usage::

    FVQA_DIR=/data/FVQA python eval_fvqa.py
    FVQA_DIR=/data/FVQA MAX_EXAMPLES=50 python eval_fvqa.py
    # Resume a partial run — already-completed data_ids are skipped automatically.
    FVQA_DIR=/data/FVQA OUTPUT_FILE=fvqa_results.jsonl python eval_fvqa.py
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from rlm import RLM, ImageContext
from rlm.logger import RLMLogger
from rlm.tools.fvqa_search import make_image_search_tool

load_dotenv()

# ---------------------------------------------------------------------------
# Config defaults (overridable via env vars or CLI)
# ---------------------------------------------------------------------------

DEFAULT_FVQA_DIR = os.environ.get("FVQA_DIR", "/home/dyvm6xra/dyvm6xrauser04/xijia/mm_zs/MMZeroSearch/mmsearch_r1/data/FVQA")
DEFAULT_MODEL = os.environ.get("RLM_MODEL_NAME", "gemini-3-flash-preview")
DEFAULT_BASE_URL = os.environ.get("RLM_BASE_URL", "https://yunwu.ai/v1")
DEFAULT_API_KEY_ENV = os.environ.get("RLM_API_KEY_ENV", "YUNWU_API_KEY")
DEFAULT_OUTPUT = os.environ.get("OUTPUT_FILE", "fvqa_results.jsonl")


# ---------------------------------------------------------------------------
# Answer normalisation & evaluation
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    s = s.strip().lower()
    for ch in [",", ".", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def is_correct(pred: str, gold: str) -> bool:
    """Exact match after normalisation (FVQA standard metric)."""
    return normalize_answer(pred) == normalize_answer(gold)


# ---------------------------------------------------------------------------
# RLM builder
# ---------------------------------------------------------------------------

def build_rlm(model_name: str, base_url: str, api_key: str, question_id: str,
              cache_path: str, log_dir: str) -> tuple[RLM, dict]:
    """Return an (rlm, image_search_tool_spec) pair configured for one sample."""
    image_search = make_image_search_tool(cache_path)
    image_search_tool_spec = {
        "tool": image_search,
        "description": (
            "image_search(question_id: str) -> SearchResultSet. "
            f"The current question ID is '{question_id}'. "
            "Returns cached web search results for the FVQA question. "
            "Print the result to see numbered web page titles and thumbnail labels. "
            "To view a specific thumbnail call result.get_thumbnail(i) — it returns "
            "a PIL Image — then pass it to view_image(thumb, 'your prompt') to "
            "inspect it visually."
        ),
    }

    logger = RLMLogger(log_dir=log_dir)

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key,
        },
        environment="local",
        max_depth=1,
        max_iterations=5,
        verbose=False,
        logger=logger,
        custom_tools={"image_search": image_search_tool_spec},
    )
    return rlm, image_search_tool_spec


# ---------------------------------------------------------------------------
# Per-sample prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TMPL = """\
You are answering a factual visual question from the FVQA benchmark.

You have access to:
  - context  : the query image (PIL Image, available as `context` in the REPL)
  - view_image(image, prompt)   : inspect any PIL Image with the vision model
  - image_search(question_id)   : retrieve cached web search results for this question
      → print the result to see titles and thumbnail labels
      → call result.get_thumbnail(i) to fetch thumbnail i as a PIL Image
      → pass the PIL Image to view_image() to inspect it

Current question ID: {question_id}
Question: {question}

Strategy:
1. Use view_image(context, "Describe the image") to understand the query image.
2. Call image_search("{question_id}") and inspect the returned titles.
3. Use view_image(result.get_thumbnail(i), "...") on the most relevant thumbnail(s).
4. Reason step-by-step, then output your final answer.

IMPORTANT: your FINAL answer must be a short factual span (a word or short phrase).
End your response with a line in exactly this format:
    ANSWER: <your answer>
"""


def extract_answer(response: str) -> str:
    """Pull out the text after the last 'ANSWER:' tag, falling back to the full response."""
    # Find the last occurrence of "ANSWER:" (case-insensitive)
    match = re.search(r"(?i)ANSWER:\s*(.+?)(?:\n|$)", response)
    if match:
        return match.group(1).strip()
    # Fall back: last non-empty line
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def load_done_ids(output_path: str) -> set[str]:
    """Return the set of data_ids already written to the output JSONL."""
    done: set[str] = set()
    p = Path(output_path)
    if not p.exists():
        return done
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "data_id" in rec:
                    done.add(rec["data_id"])
            except json.JSONDecodeError:
                pass
    return done


_write_lock = threading.Lock()


def append_result(output_path: str, record: dict[str, Any]) -> None:
    """Append one JSON record to the output file, thread-safely."""
    with _write_lock:
        with open(output_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-sample worker (runs in a thread)
# ---------------------------------------------------------------------------

def _run_one(
    question_id: str,
    question: str,
    ground_truth: str,
    query_image: Image.Image,
    model_name: str,
    base_url: str,
    api_key: str,
    cache_path: str,
    log_dir: str,
) -> dict[str, Any]:
    """Run the RLM agent on a single sample and return the result record."""
    rlm, _ = build_rlm(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        question_id=question_id,
        cache_path=cache_path,
        log_dir=log_dir,
    )
    prompt = SYSTEM_PROMPT_TMPL.format(question_id=question_id, question=question)

    pred_raw = ""
    pred = ""
    error_msg = None
    try:
        result = rlm.completion(ImageContext(query_image), root_prompt=prompt)
        pred_raw = result.response.strip()
        pred = extract_answer(pred_raw)
    except Exception as exc:
        error_msg = str(exc)

    record: dict[str, Any] = {
        "data_id": question_id,
        "question": question,
        "ground_truth": ground_truth,
        "pred_raw": pred_raw,
        "pred": pred,
        "correct": is_correct(pred, ground_truth),
    }
    if error_msg:
        record["error"] = error_msg
    return record


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_fvqa(
    fvqa_dir: str,
    model_name: str,
    base_url: str,
    api_key: str,
    output_path: str,
    start_idx: int = 0,
    end_idx: int | None = None,
    max_examples: int | None = None,
    workers: int = 8,
) -> dict[str, Any]:
    parquet_path = os.path.join(fvqa_dir, "fvqa_test.parquet")
    cache_path = os.path.join(fvqa_dir, "search_cache.pkl")
    log_dir = f"./logs/fvqa_{model_name.replace('/', '_')}"

    print(f"Loading {parquet_path} …")
    df = pd.read_parquet(parquet_path)
    print(f"Total samples: {len(df)}")

    # Slice the requested range
    df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    if max_examples is not None:
        df = df.iloc[:max_examples]
    print(f"Evaluating {len(df)} samples (start={start_idx}, end={end_idx or 'all'}, workers={workers})")

    done_ids = load_done_ids(output_path)
    if done_ids:
        print(f"Resuming: {len(done_ids)} sample(s) already done, skipping them.")

    # Build the list of pending samples (skip already-done ones)
    pending = []
    for _, row in df.iterrows():
        question_id: str = row["data_id"]
        if question_id in done_ids:
            continue
        prompt_messages = row["prompt"]
        question = (
            prompt_messages[0]["content"]
            if prompt_messages and isinstance(prompt_messages[0].get("content"), str)
            else str(prompt_messages)
        )
        ground_truth: str = row["reward_model"]["ground_truth"]
        img_entry = row["images"][0]
        query_image = Image.open(io.BytesIO(img_entry["bytes"])).convert("RGB")
        pending.append((question_id, question, ground_truth, query_image))

    print(f"Pending: {len(pending)} sample(s)")

    correct = 0
    total = 0
    t0 = time.time()

    # with ThreadPoolExecutor(max_workers=workers) as executor:
    #     futures = {
    #         executor.submit(
    #             _run_one,
    #             question_id, question, ground_truth, query_image,
    #             model_name, base_url, api_key, cache_path, log_dir,
    #         ): question_id
    #         for question_id, question, ground_truth, query_image in pending
    #     }

    #     for future in as_completed(futures):
    #         question_id = futures[future]
    #         try:
    #             record = future.result()
    #         except Exception as exc:
    #             record = {
    #                 "data_id": question_id,
    #                 "question": "",
    #                 "ground_truth": "",
    #                 "pred_raw": "",
    #                 "pred": "",
    #                 "correct": False,
    #                 "error": str(exc),
    #             }

    #         append_result(output_path, record)

    #         if record["correct"]:
    #             correct += 1
    #         total += 1

    #         elapsed = time.time() - t0
    #         avg_sec = elapsed / total
    #         remaining = (len(pending) - total) * avg_sec
    #         status = "✓" if record["correct"] else "✗"
    #         error_suffix = f"  [ERR: {record['error'][:60]}]" if record.get("error") else ""
    #         print(
    #             f"[{total}/{len(pending)}] "
    #             f"acc={correct/total:.3f}  "
    #             f"id={record['data_id']}  "
    #             f"gold='{record['ground_truth']}'  pred='{record['pred']}'  "
    #             f"{status}  "
    #             f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s"
    #             f"{error_suffix}"
    #         )
    for question_id, question, ground_truth, query_image in pending:
        record = _run_one(
            question_id, question, ground_truth, query_image,
            model_name, base_url, api_key, cache_path, log_dir,
        )
        append_result(output_path, record)
        if record["correct"]:
            correct += 1
        total += 1
        elapsed = time.time() - t0
        avg_sec = elapsed / total
        remaining = (len(pending) - total) * avg_sec
        status = "✓" if record["correct"] else "✗"
        error_suffix = f"  [ERR: {record['error'][:60]}]" if record.get("error") else ""
        print(
            f"[{total}/{len(pending)}] "
            f"acc={correct/total:.3f}  "
            f"id={record['data_id']}  "
            f"gold='{record['ground_truth']}'  pred='{record['pred']}'  "
            f"{status}  "
            f"elapsed={elapsed:.0f}s  eta={remaining:.0f}s"
            f"{error_suffix}"
        )

    # -----------------------------------------------------------------------
    # Final evaluation over the full output file (includes previous runs)
    # -----------------------------------------------------------------------
    all_correct = 0
    all_total = 0
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                all_total += 1
                if rec.get("correct"):
                    all_correct += 1
            except json.JSONDecodeError:
                pass

    accuracy = all_correct / all_total if all_total > 0 else 0.0
    summary = {
        "accuracy": accuracy,
        "correct": all_correct,
        "total": all_total,
        "output_file": output_path,
        "model": model_name,
    }
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate RLM on FVQA test set.")
    p.add_argument("--fvqa-dir", default=DEFAULT_FVQA_DIR,
                   help="Path to FVQA data directory")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Model name (default: %(default)s)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible base URL")
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
                   help="Name of env-var holding the API key")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output JSONL file path (default: %(default)s)")
    p.add_argument("--start", type=int, default=int(os.environ.get("START_IDX", 0)),
                   help="First sample index (0-based)")
    p.add_argument("--end", type=int, default=None,
                   help="One-past-last sample index")
    p.add_argument("--max-examples", type=int,
                   default=int(os.environ.get("MAX_EXAMPLES", 0)) or None,
                   help="Cap on samples to run (after slicing start/end)")
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("WORKERS", 8)),
                   help="Number of concurrent API workers (default: 8)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(
            f"Warning: environment variable '{args.api_key_env}' is not set. "
            "The API call may fail.",
            file=sys.stderr,
        )

    print("=== FVQA Evaluation ===")
    print(f"  model      : {args.model}")
    print(f"  base_url   : {args.base_url}")
    print(f"  fvqa_dir   : {args.fvqa_dir}")
    print(f"  output     : {args.output}")
    print(f"  range      : [{args.start}, {args.end or 'end'}]")
    print(f"  max_examples: {args.max_examples or 'all'}")
    print(f"  workers    : {args.workers}")
    print()

    summary = evaluate_fvqa(
        fvqa_dir=args.fvqa_dir,
        model_name=args.model,
        base_url=args.base_url,
        api_key=api_key,
        output_path=args.output,
        start_idx=args.start,
        end_idx=args.end,
        max_examples=args.max_examples,
        workers=args.workers,
    )

    print()
    print("=== Final Results ===")
    print(f"  Model   : {summary['model']}")
    print(f"  Output  : {summary['output_file']}")
    print(f"  Correct : {summary['correct']} / {summary['total']}")
    print(f"  Accuracy: {summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
