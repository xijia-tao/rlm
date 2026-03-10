"""
FVQA image search tool for use in the RLM REPL environment.

The FVQA dataset ships with a precomputed search cache (search_cache.pkl) that
maps each question ID (e.g. "fvqa_test_1717") to the top-k web search results
retrieved for that question's image.  Each entry contains:
    - tool_returned_web_title_list: list of page titles (text snippets)
    - tool_returned_images_urls:    list of thumbnail URLs

Usage::

    from rlm.tools.fvqa_search import make_image_search_tool

    image_search = make_image_search_tool("/path/to/search_cache.pkl")

    # Pass as a custom tool to RLM:
    rlm = RLM(
        ...,
        custom_tools={
            "image_search": {
                "tool": image_search,
                "description": (
                    "image_search(question_id) -> SearchResultSet. "
                    "Returns web titles and thumbnail URLs for the given FVQA question ID. "
                    "Call result.get_thumbnail(i) to fetch thumbnail i as a PIL Image for view_image()."
                ),
            }
        },
    )

    # Inside a REPL block the model can do:
    #   results = image_search("fvqa_test_1717")
    #   print(results)                          # see titles + URLs
    #   thumb = results.get_thumbnail(0)        # PIL Image
    #   desc = view_image(thumb, "What is shown?")
"""

from __future__ import annotations

import io
import pickle
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# PIL backward-compatibility shim
# ---------------------------------------------------------------------------
# search_cache.pkl was pickled with an older Pillow where JpegImageFile only
# stored ``layers`` (one value) at state[6], but Pillow ≥10 added a separate
# ``layer`` field so __setstate__ expects two values.  We patch the method
# temporarily while loading so the unpickling succeeds.

@contextmanager
def _pillow_compat_patch():
    """Context manager that patches PIL to tolerate old-style JPEG state."""
    try:
        import PIL.JpegImagePlugin as _jpeg_mod

        _orig = _jpeg_mod.JpegImageFile.__setstate__

        def _compat_setstate(self, state: list) -> None:  # noqa: ANN001
            tail = state[6:]
            if len(tail) == 1:
                self.layers = tail[0]
                self.layer = []
                # Call ImageFile.__setstate__ directly, skipping the broken one
                import PIL.ImageFile as _imgfile_mod
                _imgfile_mod.ImageFile.__setstate__(self, state)
            else:
                _orig(self, state)

        _jpeg_mod.JpegImageFile.__setstate__ = _compat_setstate
        try:
            yield
        finally:
            _jpeg_mod.JpegImageFile.__setstate__ = _orig
    except ImportError:
        yield  # PIL not available; let pickle fail naturally


# ---------------------------------------------------------------------------
# Search result container
# ---------------------------------------------------------------------------


@dataclass
class SearchResultSet:
    """Top-k search results for a single FVQA question.

    Attributes:
        question_id:  The FVQA question identifier (e.g. "fvqa_test_1717").
        titles:       Ordered list of web-page titles (text snippets).
        image_urls:   Ordered list of thumbnail image URLs (same length as titles).

    Methods:
        get_thumbnail(index)  Download thumbnail *index* and return a PIL Image.
        __len__               Number of results.
        __str__ / __repr__    Human-readable summary suitable for REPL print().
    """

    question_id: str
    titles: list[str] = field(default_factory=list)
    image_urls: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Thumbnail fetching
    # ------------------------------------------------------------------

    def get_thumbnail(self, index: int, timeout: int = 10) -> Any:
        """Return thumbnail *index* as a PIL Image.

        Each thumbnail is either a URL string (fetched on demand) or a PIL
        Image already stored in the cache (returned directly).  The result
        is always a PIL Image in RGB mode and can be passed straight to
        ``view_image()``::

            thumb = results.get_thumbnail(0)
            answer = view_image(thumb, "What is shown in this image?")

        Args:
            index:   Zero-based index into the result list.
            timeout: HTTP request timeout in seconds when fetching a URL
                     (ignored for pre-cached PIL Images).

        Returns:
            A PIL ``Image`` object (RGB mode).

        Raises:
            IndexError:  If *index* is out of range.
            RuntimeError: If the download fails or the URL is not a valid image.
        """
        from PIL import Image

        if index < 0 or index >= len(self.image_urls):
            raise IndexError(
                f"Index {index} out of range — this result set has {len(self.image_urls)} items."
            )

        entry = self.image_urls[index]

        # Already a PIL Image (stored directly in the cache)
        if isinstance(entry, Image.Image):
            return entry.convert("RGB")

        # URL string — download it
        url = str(entry)
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (X11; Linux x86_64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0 Safari/537.36"
                    )
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Failed to download thumbnail {index} from {url!r}: {exc}"
            ) from exc

        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as exc:
            raise RuntimeError(
                f"Downloaded data from {url!r} is not a valid image: {exc}"
            ) from exc

        return img

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _thumbnail_label(self, entry: Any) -> str:
        """Return a short human-readable label for a thumbnail entry."""
        try:
            from PIL import Image as _PILImage
            if isinstance(entry, _PILImage.Image):
                return f"<cached image {entry.size[0]}x{entry.size[1]}>"
        except ImportError:
            pass
        return str(entry)

    def __len__(self) -> int:
        return len(self.titles)

    def __str__(self) -> str:
        if not self.titles:
            return f"SearchResultSet(question_id={self.question_id!r}, no results)"
        lines = [f"Search results for '{self.question_id}' ({len(self)} result(s)):"]
        for i, (title, url_entry) in enumerate(zip(self.titles, self.image_urls)):
            lines.append(f"  [{i}] Title : {title}")
            lines.append(f"       Image  : {self._thumbnail_label(url_entry)}")
        lines.append(
            "\nTo view a thumbnail: thumb = results.get_thumbnail(i); view_image(thumb, 'describe')"
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SearchResultSet(question_id={self.question_id!r}, "
            f"n_results={len(self)})"
        )


# ---------------------------------------------------------------------------
# Cache wrapper
# ---------------------------------------------------------------------------


class FVQASearchCache:
    """Lazy-loading wrapper around the FVQA ``search_cache.pkl`` file.

    The pickle maps question IDs to dicts with keys:
        - ``tool_returned_web_title_list``
        - ``tool_returned_images_urls``

    Example::

        cache = FVQASearchCache("/path/to/search_cache.pkl")
        results = cache.search("fvqa_test_1717")
        print(results)
        thumb = results.get_thumbnail(0)

    The cache is loaded into memory on first access (lazy init).
    """

    def __init__(self, cache_path: str | Path):
        self._cache_path = Path(cache_path)
        self._data: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._data is None:
            with _pillow_compat_patch(), open(self._cache_path, "rb") as fh:
                self._data = pickle.load(fh)

    @property
    def data(self) -> dict[str, Any]:
        self._load()
        return self._data  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, question_id: str) -> SearchResultSet:
        """Return the cached search results for *question_id*.

        Args:
            question_id: FVQA question identifier, e.g. ``"fvqa_test_1717"``.

        Returns:
            A :class:`SearchResultSet` containing web titles and thumbnail URLs.

        Raises:
            KeyError: If *question_id* is not present in the cache.
        """
        if question_id not in self.data:
            sample = list(self.data.keys())[:5]
            raise KeyError(
                f"No cached results for '{question_id}'. "
                f"Sample of available IDs: {sample}"
            )
        entry = self.data[question_id]
        return SearchResultSet(
            question_id=question_id,
            titles=list(entry.get("tool_returned_web_title_list", [])),
            image_urls=list(entry.get("tool_returned_images_urls", [])),
        )

    def __contains__(self, question_id: str) -> bool:
        return question_id in self.data

    def __len__(self) -> int:
        return len(self.data)

    def keys(self):
        return self.data.keys()

    def __repr__(self) -> str:
        loaded = "loaded" if self._data is not None else "not yet loaded"
        return f"FVQASearchCache({self._cache_path!r}, {loaded})"


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_image_search_tool(cache_path: str | Path) -> callable:
    """Create an ``image_search`` callable backed by the FVQA search cache.

    The returned function has the signature::

        image_search(question_id: str) -> SearchResultSet

    and is designed to be passed as a ``custom_tool`` to :class:`~rlm.RLM`::

        from rlm.tools.fvqa_search import make_image_search_tool

        image_search = make_image_search_tool("/path/to/search_cache.pkl")

        rlm = RLM(
            ...,
            custom_tools={
                "image_search": {
                    "tool": image_search,
                    "description": (
                        "image_search(question_id) -> SearchResultSet. "
                        "Returns cached web search results for the FVQA question. "
                        "Print the result to see titles and thumbnail URLs. "
                        "Call result.get_thumbnail(i) to fetch thumbnail i as a "
                        "PIL Image, then pass it to view_image() to inspect it."
                    ),
                }
            },
        )

    Args:
        cache_path: Path to ``search_cache.pkl``.

    Returns:
        A callable ``image_search(question_id) -> SearchResultSet``.
    """
    cache = FVQASearchCache(cache_path)

    def image_search(question_id: str) -> SearchResultSet:
        """Return cached web search results for a FVQA question ID.

        Args:
            question_id: E.g. ``"fvqa_test_1717"``.

        Returns:
            :class:`SearchResultSet` — print it to see titles and thumbnail URLs.
            Call ``.get_thumbnail(i)`` to fetch thumbnail *i* as a PIL Image for
            ``view_image()``.
        """
        return cache.search(question_id)

    return image_search
