"""
FVQA VQA example: answering visual questions with on-demand image search.

Each FVQA sample contains a query image and a factual question.  Pre-computed
web search results (titles + thumbnail URLs) are cached in a .pkl file keyed
by question ID.  This example injects an ``image_search`` custom tool so that
the model can:

  1. Inspect the query image with ``view_image()``.
  2. Call ``image_search(question_id)`` to retrieve web search results.
  3. Selectively fetch and view individual thumbnail images via
     ``result.get_thumbnail(i)`` + ``view_image()`` — rather than having all
     thumbnails pushed into the context at once.

Data layout::

    /path/to/FVQA/
        fvqa_test.parquet   — rows: prompt, images (bytes), data_id, reward_model
        search_cache.pkl    — dict[data_id -> {titles, image_urls}]

Requirements::

    pip install pillow pandas pyarrow
    export OPENAI_API_KEY=...   (or set in .env)

The model used must support vision (e.g. gpt-4o, gpt-4o-mini, Qwen2.5-VL-*).
"""

import io
import os

import pandas as pd
from dotenv import load_dotenv
from PIL import Image

from rlm import RLM, ImageContext
from rlm.tools.fvqa_search import make_image_search_tool

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FVQA_DIR = "/home/dyvm6xra/dyvm6xrauser04/xijia/mm_zs/MMZeroSearch/mmsearch_r1/data/FVQA"
PARQUET_PATH = os.path.join(FVQA_DIR, "fvqa_test.parquet")
CACHE_PATH = os.path.join(FVQA_DIR, "search_cache.pkl")

# Index of the FVQA test sample to evaluate (change to try other questions)
SAMPLE_INDEX = 0

# ---------------------------------------------------------------------------
# Load a single FVQA sample
# ---------------------------------------------------------------------------

df = pd.read_parquet(PARQUET_PATH)
row = df.iloc[SAMPLE_INDEX]

# Extract question text
prompt_messages = row["prompt"]
question = prompt_messages[0]["content"] if len(prompt_messages) > 0 else "Describe this image."

# Extract the query image (stored as raw bytes in the parquet)
img_entry = row["images"][0]
query_image = Image.open(io.BytesIO(img_entry["bytes"])).convert("RGB")

# Question ID — used to look up the search cache
question_id: str = row["data_id"]

# Ground truth (for reference / evaluation)
ground_truth: str = row["reward_model"]["ground_truth"]

print(f"Question ID : {question_id}")
print(f"Question    : {question}")
print(f"Ground truth: {ground_truth}")
print(f"Image size  : {query_image.size}")

# ---------------------------------------------------------------------------
# Build the image_search custom tool
# ---------------------------------------------------------------------------

image_search = make_image_search_tool(CACHE_PATH)

# The description is shown in the model's system prompt so it knows what the
# tool returns and how to use get_thumbnail() + view_image().
image_search_tool_spec = {
    "tool": image_search,
    "description": (
        "image_search(question_id: str) -> SearchResultSet. "
        f"The current question ID is '{question_id}'. "
        "Returns cached web search results for a FVQA question. "
        "Print the result to see numbered web page titles and thumbnail URLs. "
        "To view a specific thumbnail as an image call "
        "result.get_thumbnail(i) which returns a PIL Image — pass it to "
        "view_image(thumb, 'your prompt') to inspect it visually."
    ),
}

# ---------------------------------------------------------------------------
# Run the RLM
# ---------------------------------------------------------------------------

# rlm = RLM(
#     backend="openai",
#     backend_kwargs={
#         "model_name": "Qwen/Qwen3-VL-8B-Instruct",
#         "base_url": "http://localhost:11434/v1",
#         "api_key": "openai",
#     },
#     environment="local",
#     max_depth=1,
#     max_iterations=10,
#     verbose=True,
#     custom_tools={"image_search": image_search_tool_spec},
# )

rlm = RLM(
    backend="openai",
    backend_kwargs={
        "model_name": "gemini-3-flash-preview", # "gemini-3.1-pro-preview",
        "base_url": "https://yunwu.ai/v1",
        "api_key": os.environ.get("YUNWU_API_KEY"),
    },
    environment="local",
    max_depth=1,
    max_iterations=5,
    verbose=True,
    custom_tools={"image_search": image_search_tool_spec},
)

# The query image is the context; the question is the root_prompt.
# The model will call view_image() to inspect the query image, then optionally
# call image_search() and get_thumbnail() to examine search result thumbnails.
result = rlm.completion(
    ImageContext(query_image),
    root_prompt=question,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print("\n=== Final Answer ===")
print(result.response)
print(f"\n=== Ground Truth ===")
print(ground_truth)
