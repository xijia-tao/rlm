from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

ClientBackend = Literal[
    "openai",
    "portkey",
    "openrouter",
    "vercel",
    "vllm",
    "litellm",
    "anthropic",
    "azure_openai",
    "gemini",
]
EnvironmentType = Literal["local", "docker", "modal", "prime", "daytona", "e2b"]


def _serialize_pil_image(value: Any) -> dict | None:
    """Serialize a PIL Image to a base64 JSON dict. Returns None if not a PIL Image."""
    try:
        from PIL import Image

        if not isinstance(value, Image.Image):
            return None
    except ImportError:
        return None

    import base64
    import io

    original_size = [value.width, value.height]
    img = value.copy()
    # Downsample large images to keep log sizes reasonable
    if img.width > 512 or img.height > 512:
        img.thumbnail((512, 512), Image.LANCZOS)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {
        "__type__": "pil_image",
        "data": b64,
        "format": "jpeg",
        "mode": value.mode,
        "size": original_size,
    }


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, ModuleType):
        return f"<module '{value.__name__}'>"
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if callable(value):
        return f"<{type(value).__name__} '{getattr(value, '__name__', repr(value))}'>"
    # Detect PIL Images and serialize as base64
    pil_result = _serialize_pil_image(value)
    if pil_result is not None:
        return pil_result
    # Try to convert to string for other types
    try:
        return repr(value)
    except Exception:
        return f"<{type(value).__name__}>"


########################################################
########    Types for LM Cost Tracking         #########
########################################################


@dataclass
class ModelUsageSummary:
    total_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float | None = None  # Cost in USD, if available from provider

    def to_dict(self):
        result = {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ModelUsageSummary":
        return cls(
            total_calls=data.get("total_calls"),
            total_input_tokens=data.get("total_input_tokens"),
            total_output_tokens=data.get("total_output_tokens"),
            total_cost=data.get("total_cost"),
        )


@dataclass
class UsageSummary:
    model_usage_summaries: dict[str, ModelUsageSummary]

    @property
    def total_cost(self) -> float | None:
        """Aggregate cost across all models. Returns None if no cost data available."""
        costs = [
            summary.total_cost
            for summary in self.model_usage_summaries.values()
            if summary.total_cost is not None
        ]
        return sum(costs) if costs else None

    @property
    def total_input_tokens(self) -> int:
        """Aggregate input tokens across all models."""
        return sum(summary.total_input_tokens for summary in self.model_usage_summaries.values())

    @property
    def total_output_tokens(self) -> int:
        """Aggregate output tokens across all models."""
        return sum(summary.total_output_tokens for summary in self.model_usage_summaries.values())

    def to_dict(self):
        result = {
            "model_usage_summaries": {
                model: usage_summary.to_dict()
                for model, usage_summary in self.model_usage_summaries.items()
            },
        }
        if self.total_cost is not None:
            result["total_cost"] = self.total_cost
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "UsageSummary":
        return cls(
            model_usage_summaries={
                model: ModelUsageSummary.from_dict(usage_summary)
                for model, usage_summary in data.get("model_usage_summaries", {}).items()
            },
        )


########################################################
########   Types for REPL and RLM Iterations   #########
########################################################
@dataclass
class RLMChatCompletion:
    """Record of a single LLM call made from within the environment."""

    root_model: str
    prompt: str | dict[str, Any]
    response: str
    usage_summary: UsageSummary
    execution_time: float
    metadata: dict | None = (
        None  # Full trajectory (run_metadata + iterations) when logger captures it
    )

    def to_dict(self):
        out = {
            "root_model": self.root_model,
            "prompt": self.prompt,
            "response": self.response,
            "usage_summary": self.usage_summary.to_dict(),
            "execution_time": self.execution_time,
        }
        if self.metadata is not None:
            out["metadata"] = self.metadata
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "RLMChatCompletion":
        return cls(
            root_model=data.get("root_model"),
            prompt=data.get("prompt"),
            response=data.get("response"),
            usage_summary=UsageSummary.from_dict(data.get("usage_summary")),
            execution_time=data.get("execution_time"),
            metadata=data.get("metadata"),
        )


@dataclass
class REPLResult:
    stdout: str
    stderr: str
    locals: dict
    execution_time: float
    llm_calls: list["RLMChatCompletion"]
    final_answer: str | None = None

    def __init__(
        self,
        stdout: str,
        stderr: str,
        locals: dict,
        execution_time: float = None,
        rlm_calls: list["RLMChatCompletion"] = None,
        final_answer: str | None = None,
    ):
        self.stdout = stdout
        self.stderr = stderr
        self.locals = locals
        self.execution_time = execution_time
        self.rlm_calls = rlm_calls or []
        self.final_answer = final_answer

    def __str__(self):
        return f"REPLResult(stdout={self.stdout}, stderr={self.stderr}, locals={self.locals}, execution_time={self.execution_time}, rlm_calls={len(self.rlm_calls)})"

    def to_dict(self):
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "locals": {k: _serialize_value(v) for k, v in self.locals.items()},
            "execution_time": self.execution_time,
            "rlm_calls": [call.to_dict() for call in self.rlm_calls],
            "final_answer": self.final_answer,
        }


@dataclass
class CodeBlock:
    code: str
    result: REPLResult

    def to_dict(self):
        return {"code": self.code, "result": self.result.to_dict()}


@dataclass
class RLMIteration:
    prompt: str | dict[str, Any]
    response: str
    code_blocks: list[CodeBlock]
    final_answer: str | None = None
    iteration_time: float | None = None

    def to_dict(self):
        return {
            "prompt": self.prompt,
            "response": self.response,
            "code_blocks": [code_block.to_dict() for code_block in self.code_blocks],
            "final_answer": self.final_answer,
            "iteration_time": self.iteration_time,
        }


########################################################
########   Types for RLM Metadata   #########
########################################################


@dataclass
class RLMMetadata:
    """Metadata about the RLM configuration."""

    root_model: str
    max_depth: int
    max_iterations: int
    backend: str
    backend_kwargs: dict[str, Any]
    environment_type: str
    environment_kwargs: dict[str, Any]
    other_backends: list[str] | None = None

    def to_dict(self):
        return {
            "root_model": self.root_model,
            "max_depth": self.max_depth,
            "max_iterations": self.max_iterations,
            "backend": self.backend,
            "backend_kwargs": {k: _serialize_value(v) for k, v in self.backend_kwargs.items()},
            "environment_type": self.environment_type,
            "environment_kwargs": {
                k: _serialize_value(v) for k, v in self.environment_kwargs.items()
            },
            "other_backends": self.other_backends,
        }


########################################################
########   Types for RLM Prompting   #########
########################################################


@dataclass
class VideoMetadata:
    """Metadata for a video file."""

    duration_sec: float
    fps: float
    width: int
    height: int
    total_frames: int

    @property
    def duration_str(self) -> str:
        minutes = int(self.duration_sec // 60)
        seconds = self.duration_sec % 60
        return f"{minutes}m{seconds:.1f}s"

    def __repr__(self) -> str:
        return (
            f"VideoMetadata(duration={self.duration_str}, fps={self.fps:.2f}, "
            f"resolution={self.width}x{self.height}, total_frames={self.total_frames})"
        )


class VideoContext:
    """
    Wraps a video file to pass as context to an RLM.

    The model can inspect the video on demand by sampling or seeking to specific
    frames, then calling ``view_image()`` to visually examine them.

    Usage::

        from rlm import RLM, VideoContext
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-4o"}, ...)
        result = rlm.completion(
            VideoContext("video.mp4"),
            root_prompt="What happens at the beginning of this video?",
        )

    Requires ``opencv-python``::

        pip install opencv-python
    """

    def __init__(self, source: str | Path):
        """
        Args:
            source: File path to a video file (str or Path).
        """
        self.source = Path(source)
        self._metadata: VideoMetadata | None = None

    def _open_cap(self) -> tuple[Any, Any]:
        """Open a cv2 VideoCapture. Caller must call cap.release()."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for VideoContext. "
                "Install it with: pip install opencv-python"
            ) from None
        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.source}")
        return cap, cv2

    @property
    def metadata(self) -> "VideoMetadata":
        """Return video metadata (lazy-loaded and cached)."""
        if self._metadata is None:
            cap, cv2 = self._open_cap()
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration_sec = total_frames / fps if fps > 0 else 0.0
                self._metadata = VideoMetadata(
                    duration_sec=duration_sec,
                    fps=fps,
                    width=width,
                    height=height,
                    total_frames=total_frames,
                )
            finally:
                cap.release()
        return self._metadata

    def get_frame(self, timestamp_sec: float) -> Any:
        """
        Return a PIL Image of the frame at the given timestamp.

        Args:
            timestamp_sec: Time in seconds. Clamped to [0, duration].

        Returns:
            PIL Image (RGB).
        """
        from PIL import Image

        cap, cv2 = self._open_cap()
        try:
            meta = self.metadata
            timestamp_sec = max(0.0, min(timestamp_sec, meta.duration_sec))
            frame_idx = int(timestamp_sec * meta.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(
                    f"Could not read frame at {timestamp_sec:.2f}s (index {frame_idx})"
                )
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

    def get_frames(self, timestamps: list[float]) -> list[Any]:
        """
        Return PIL Images for each of the given timestamps.

        Keeps the video capture open across all seeks, which is more efficient
        than calling ``get_frame()`` in a loop.

        Args:
            timestamps: List of times in seconds.

        Returns:
            List of PIL Images in the same order as input.
        """
        from PIL import Image

        cap, cv2 = self._open_cap()
        try:
            meta = self.metadata
            results = []
            for t in timestamps:
                t = max(0.0, min(t, meta.duration_sec))
                frame_idx = int(t * meta.fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(
                        f"Could not read frame at {t:.2f}s (index {frame_idx})"
                    )
                results.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return results
        finally:
            cap.release()

    def sample_frames(self, n: int = 8) -> list[tuple[float, Any]]:
        """
        Uniformly sample ``n`` frames from the video.

        Timestamps are placed at the midpoints of ``n`` equal-length segments,
        avoiding boundary frames that may be black or incomplete.

        Args:
            n: Number of frames to sample (default 8).

        Returns:
            List of ``(timestamp_sec, PIL Image)`` tuples.
        """
        if n <= 0:
            return []
        meta = self.metadata
        # Midpoints of n equal segments: t_i = (2i+1) / (2n) * duration
        timestamps = [meta.duration_sec * (2 * i + 1) / (2 * n) for i in range(n)]
        frames = self.get_frames(timestamps)
        return list(zip(timestamps, frames, strict=True))

    def get_frame_at_index(self, index: int) -> Any:
        """
        Return a PIL Image of the frame at the given 0-based frame index.

        Args:
            index: Frame index. Clamped to [0, total_frames - 1].

        Returns:
            PIL Image (RGB).
        """
        meta = self.metadata
        index = max(0, min(index, meta.total_frames - 1))
        return self.get_frame(index / meta.fps)

    def get_source_path(self) -> str:
        """Return the file path as a string."""
        return str(self.source)

    def __repr__(self) -> str:
        return f"VideoContext({str(self.source)!r})"


class ImageContext:
    """
    Wraps an image (file path or PIL Image) to pass as context to an RLM.

    Usage::

        from rlm import RLM, ImageContext
        rlm = RLM(backend="openai", backend_kwargs={"model_name": "gpt-4o"}, ...)
        result = rlm.completion(ImageContext("photo.jpg"), root_prompt="What is in this image?")
    """

    def __init__(self, source: str | Path | Any):
        """
        Args:
            source: File path to an image (str or Path) or a PIL Image object.
        """
        self.source = source

    def load(self) -> Any:
        """Return the image as a PIL Image object."""
        from rlm.utils.image_utils import load_image

        return load_image(self.source)

    def encode_base64(self, fmt: str = "PNG") -> str:
        """Encode the image as a base64 string."""
        from rlm.utils.image_utils import encode_image_base64

        return encode_image_base64(self.source, fmt=fmt)

    def get_source_path(self) -> str | None:
        """Return the file path if source is a path, otherwise None."""
        if isinstance(self.source, (str, Path)):
            return str(self.source)
        return None

    def __repr__(self) -> str:
        src = self.get_source_path() or "<PIL Image>"
        return f"ImageContext({src!r})"


@dataclass
class QueryMetadata:
    context_lengths: list[int]
    context_total_length: int
    context_type: str

    def __init__(
        self,
        prompt: "str | list[str] | dict[Any, Any] | list[dict[Any, Any]] | ImageContext | VideoContext",
    ):
        if isinstance(prompt, VideoContext):
            self.context_type = "video"
            self.context_lengths = [0]
            self.context_total_length = 0
        elif isinstance(prompt, ImageContext):
            self.context_type = "image"
            self.context_lengths = [0]
            self.context_total_length = 0
        elif isinstance(prompt, str):
            self.context_lengths = [len(prompt)]
            self.context_type = "str"
        elif isinstance(prompt, dict):
            self.context_type = "dict"
            self.context_lengths = []
            for chunk in prompt.values():
                if isinstance(chunk, str):
                    self.context_lengths.append(len(chunk))
                    continue
                try:
                    import json

                    self.context_lengths.append(len(json.dumps(chunk, default=str)))
                except Exception:
                    self.context_lengths.append(len(repr(chunk)))
            self.context_type = "dict"
        elif isinstance(prompt, list):
            self.context_type = "list"
            if len(prompt) == 0:
                self.context_lengths = [0]
            elif isinstance(prompt[0], dict):
                if "content" in prompt[0]:
                    self.context_lengths = [len(str(chunk.get("content", ""))) for chunk in prompt]
                else:
                    self.context_lengths = []
                    for chunk in prompt:
                        try:
                            import json

                            self.context_lengths.append(len(json.dumps(chunk, default=str)))
                        except Exception:
                            self.context_lengths.append(len(repr(chunk)))
            else:
                self.context_lengths = [len(chunk) for chunk in prompt]
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")

        self.context_total_length = sum(self.context_lengths)
