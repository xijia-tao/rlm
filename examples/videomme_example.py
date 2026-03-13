"""
VideoMME example: answering multiple-choice questions about videos.

VideoMME (https://video-mme.github.io) is a benchmark of multiple-choice
questions spanning short, medium, and long videos.  Each sample has:
  - A video file (.mp4)
  - A question with four choices (A/B/C/D)
  - A ground-truth answer

The RLM uses ``VideoContext`` to inspect the video on demand:
  1. Check metadata (duration, fps, resolution).
  2. Survey with uniformly sampled frames via ``context.sample_frames(n)``.
  3. Zoom into specific moments with ``context.get_frame(t)``.
  4. Call ``view_image(frame, prompt)`` to visually inspect any frame.

Data layout::

    /path/to/VideoMME/
        videos/
            <video_id>.mp4
            ...
        videomme.json        -- list of {video_id, question, options, answer}
        # or use the HuggingFace parquet split directly

Requirements::

    pip install "rlms[video]" pillow
    # opencv-python is pulled in by the [video] extra
    export OPENAI_API_KEY=...   (or set in .env)

The model must support vision (e.g. gpt-4o, gpt-4o-mini, Qwen2.5-VL-72B).
"""

import json
import os
import pandas as pd

from dotenv import load_dotenv

from rlm import RLM, VideoContext

load_dotenv()

# ---------------------------------------------------------------------------
# Config — edit these paths for your local VideoMME copy
# ---------------------------------------------------------------------------

VIDEOMME_DIR = "/path/to/VideoMME"
VIDEO_DIR = os.path.join(VIDEOMME_DIR, "videos")
ANNOTATION_FILE = "/home/dyvm6xra/dyvm6xrauser04/lustre/xijia/video_benchmark/Video-MME/videomme/test-00000-of-00001.parquet"


# Model (must support vision)
MODEL_NAME = "gpt-4o"

# ---------------------------------------------------------------------------
# Load a single VideoMME sample
# ---------------------------------------------------------------------------

# load parquet
annotations = pd.read_parquet(ANNOTATION_FILE)

video_id = "08km9Yqbt-A"
sample = annotations[annotations["videoID"] == video_id].iloc[0]

question: str = sample["question"]
options: list[str] = sample["options"]          # ["A. ...", "B. ...", "C. ...", "D. ..."]
ground_truth: str = sample["answer"]            # "A", "B", "C", or "D"
duration_category: str = sample.get("duration", "unknown")  # "short" / "medium" / "long"

video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

# hardcode for debug
video_path = "/home/dyvm6xra/dyvm6xrauser04/lustre/xijia/video_benchmark/Video-MME/videos_part1/data/08km9Yqbt-A.mp4"

print(f"Video ID       : {video_id}")
print(f"Duration cat.  : {duration_category}")
print(f"Video path     : {video_path}")
print(f"Question       : {question}")
for opt in options:
    print(f"  {opt}")
print(f"Ground truth   : {ground_truth}")

# ---------------------------------------------------------------------------
# Build a root prompt that includes the choices
# ---------------------------------------------------------------------------

options_str = "\n".join(options)
root_prompt = (
    f"{question}\n\nChoices:\n{options_str}\n\n"
    "Watch the video carefully, then answer with just the letter (A, B, C, or D)."
)

# ---------------------------------------------------------------------------
# Run the RLM
# ---------------------------------------------------------------------------

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "base_url": "http://localhost:11434/v1",
        "api_key": "openai",
    },
    environment="local",
    max_depth=1,
    max_iterations=5,
    verbose=True,
)

# rlm = RLM(
#     backend="openai",
#     backend_kwargs={
#         "model_name": "gemini-3-flash-preview", # "gemini-3.1-pro-preview",
#         "base_url": "https://yunwu.ai/v1",
#         "api_key": os.environ.get("YUNWU_API_KEY"),
#     },
#     environment="local",
#     max_depth=1,
#     max_iterations=5,
#     verbose=True
# )

# VideoContext wraps the .mp4 file.  The model will call .sample_frames(),
# .get_frame(), and view_image() from within REPL blocks.
result = rlm.completion(
    VideoContext(video_path),
    root_prompt=root_prompt,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("\n=== Final Answer ===")
print(result.response)
print("\n=== Ground Truth ===")
print(ground_truth)
print("\n=== Correct? ===")
print(ground_truth.upper() in result.response.upper())
