"""
Logger for RLM iterations.

Captures run metadata and iterations in memory so they can be attached to
RLMChatCompletion.metadata. Optionally writes the same data to JSON-lines files.
"""

import base64
import io
import json
import os
import uuid
from datetime import datetime
from typing import Any

from rlm.core.types import RLMIteration, RLMMetadata


class RLMLogger:
    """
    Captures trajectory (run metadata + iterations) for each completion.
    By default only captures in memory; set log_dir to also save to disk.

    - log_dir=None: trajectory is available via get_trajectory() and can be
      attached to RLMChatCompletion.metadata (no disk write).
    - log_dir="path": same capture plus appends to a JSONL file per run.
    """

    def __init__(self, log_dir: str | None = None, file_name: str = "rlm"):
        self._save_to_disk = log_dir is not None
        self.log_dir = log_dir
        self.log_file_path: str | None = None
        if self._save_to_disk and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_id = str(uuid.uuid4())[:8]
            self.log_file_path = os.path.join(log_dir, f"{file_name}_{timestamp}_{run_id}.jsonl")

        self._run_metadata: dict | None = None
        self._context_data: dict | None = None
        self._iterations: list[dict] = []
        self._iteration_count = 0
        self._metadata_logged = False

    def log_metadata(self, metadata: RLMMetadata) -> None:
        """Capture run metadata (and optionally write to file)."""
        if self._metadata_logged:
            return

        self._run_metadata = metadata.to_dict()
        self._metadata_logged = True

        if self._save_to_disk and self.log_file_path:
            entry = {
                "type": "metadata",
                "timestamp": datetime.now().isoformat(),
                **self._run_metadata,
            }
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def log_context(self, context_payload: Any) -> None:
        """Capture the input context (ImageContext/VideoContext) for visualization."""
        from rlm.core.types import ImageContext, VideoContext

        entry: dict | None = None

        if isinstance(context_payload, ImageContext):
            try:
                img = context_payload.load()
                thumb = img.copy()
                thumb.thumbnail((800, 800))
                if thumb.mode not in ("RGB", "L"):
                    thumb = thumb.convert("RGB")
                buf = io.BytesIO()
                thumb.save(buf, format="JPEG", quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                source_path = context_payload.get_source_path()
                entry = {
                    "type": "context",
                    "context_type": "image",
                    "image_data": b64,
                    "source_path": source_path,
                }
            except Exception:
                return

        elif isinstance(context_payload, VideoContext):
            try:
                samples = context_payload.sample_frames(n=6)
                frames = []
                for ts, img in samples:
                    thumb = img.copy()
                    thumb.thumbnail((400, 400))
                    if thumb.mode not in ("RGB", "L"):
                        thumb = thumb.convert("RGB")
                    buf = io.BytesIO()
                    thumb.save(buf, format="JPEG", quality=80)
                    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    frames.append({"timestamp": round(ts, 2), "data": b64})
                meta = context_payload.metadata
                entry = {
                    "type": "context",
                    "context_type": "video",
                    "frames": frames,
                    "source_path": context_payload.get_source_path(),
                    "duration_sec": meta.duration_sec,
                    "fps": meta.fps,
                    "resolution": [meta.width, meta.height],
                }
            except Exception:
                return

        if entry is None:
            return

        self._context_data = entry
        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump({"timestamp": datetime.now().isoformat(), **entry}, f)
                f.write("\n")

    def log(self, iteration: RLMIteration) -> None:
        """Capture one iteration (and optionally append to file)."""
        self._iteration_count += 1
        entry = {
            "type": "iteration",
            "iteration": self._iteration_count,
            "timestamp": datetime.now().isoformat(),
            **iteration.to_dict(),
        }
        self._iterations.append(entry)

        if self._save_to_disk and self.log_file_path:
            with open(self.log_file_path, "a") as f:
                json.dump(entry, f)
                f.write("\n")

    def clear_iterations(self) -> None:
        """Reset iterations for the next completion (trajectory is per completion)."""
        self._iterations = []
        self._iteration_count = 0

    def get_trajectory(self) -> dict | None:
        """Return captured run_metadata + iterations for the current completion, or None if no metadata yet."""
        if self._run_metadata is None:
            return None
        result: dict = {
            "run_metadata": self._run_metadata,
            "iterations": list(self._iterations),
        }
        if self._context_data is not None:
            result["context"] = self._context_data
        return result

    @property
    def iteration_count(self) -> int:
        return self._iteration_count
