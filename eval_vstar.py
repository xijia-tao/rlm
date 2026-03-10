"""
Evaluate RLM with Qwen3-VL on lmms-lab/vstar-bench.

Requirements:
    pip install datasets pillow python-dotenv
    export OPENAI_API_KEY=...          # or set in .env
    # Optional overrides:
    # export RLM_MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"
    # export RLM_BASE_URL="http://localhost:11434/v1"
    # export VSTAR_SPLIT="test"
    # export MAX_EXAMPLES="100"

This script assumes the vstar-bench dataset has columns:
    - image: PIL.Image
    - text: question
    - label: answer
"""

import os
import tempfile
from typing import Tuple

from dotenv import load_dotenv
from datasets import load_dataset
from PIL import Image

from rlm import RLM, ImageContext
from rlm.logger import RLMLogger

load_dotenv()


def normalize_answer(s: str) -> str:
    s = s.strip().lower()
    # Basic punctuation and whitespace normalization
    for ch in [",", ".", "!", "?", ":", ";", '"', "'", "(", ")", "[", "]"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def save_image_to_temp(img: Image.Image, tmp_dir: str, idx: int) -> str:
    path = os.path.join(tmp_dir, f"vstar_{idx}.png")
    img.save(path)
    return path


def build_rlm() -> RLM:
    model_name = os.environ.get("RLM_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
    base_url = os.environ.get("RLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "openai")

    # model_name = 'gpt-5.2'
    # model_name = 'gemini-3.1-pro-preview'
    # base_url = 'https://yunwu.ai/v1'
    # api_key = os.environ.get("YUNWU_API_KEY")

    logger = RLMLogger(log_dir=f"./logs/{model_name}_vstar")

    rlm = RLM(
        backend="openai",
        backend_kwargs={
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key,
        },
        environment="local",
        max_depth=1,
        max_iterations=10,
        verbose=True,
        logger=logger,
    )
    return rlm


def evaluate_vstar(
    split: str = "test",
    max_examples: int | None = None,
) -> Tuple[float, int, int]:
    print(f"Loading lmms-lab/vstar-bench (split={split})...")
    ds = load_dataset("lmms-lab/vstar-bench", split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    print(f"Dataset size: {len(ds)} examples")

    rlm = build_rlm()
    tmp_dir = tempfile.mkdtemp(prefix="vstar_bench_")

    correct = 0
    total = 0

    for idx, ex in enumerate(ds):
        img: Image.Image = ex["image"]
        question: str = ex["text"]
        gold: str = str(ex["label"])

        img_path = save_image_to_temp(img, tmp_dir, idx)
        context = ImageContext(img_path)

        prompt = (
            "You are evaluating on the V* (V-star) vision-language benchmark.\n"
            "You are given an image (accessible via the `context` object) and a question.\n"
            "Reason carefully if needed, but in your final answer, output ONLY the final answer "
            "as a short span (no explanations, no extra words).\n\n"
            f"Question: {question}\n"
        )

        try:
            result = rlm.completion(
                context,
                root_prompt=prompt,
            )
            pred = result.response.strip()
        except Exception as e:
            print(f"[{idx}] Error during completion: {e}")
            pred = ""

        total += 1
        if normalize_answer(pred) == normalize_answer(gold):
            correct += 1

        if idx % 20 == 0:
            print(
                f"[{idx}/{len(ds)}] "
                f"running acc={correct/total:.3f} "
                f"gold='{gold}' pred='{pred}'"
            )

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def main() -> None:
    split = os.environ.get("VSTAR_SPLIT", "test")
    max_examples_env = os.environ.get("MAX_EXAMPLES")
    max_examples = int(max_examples_env) if max_examples_env else None

    acc, correct, total = evaluate_vstar(split=split, max_examples=max_examples)
    print("\n=== V* Bench Evaluation (Qwen3-VL + RLM) ===")
    print(f"Split: {split}")
    if max_examples is not None:
        print(f"Max examples: {max_examples}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()