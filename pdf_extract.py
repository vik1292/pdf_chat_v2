"""PDF text extraction with coordinates.

pdfplumber is the single source of truth: chunk text is built from the same word
list that supplies highlight coordinates, so a chunk maps back to its words by index.

Coordinates are in PDF points with a top-left origin relative to the page cropbox,
which is the space streamlit-pdf-viewer expects for annotations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

import pdfplumber

BBox = Tuple[float, float, float, float]  # (x0, top, x1, bottom)

_SENTENCE_END = re.compile(r"[.!?]['\")\]]*$")


@dataclass
class Word:
    text: str
    x0: float
    top: float
    x1: float
    bottom: float
    line: int = -1


@dataclass
class PageText:
    page: int  # 0-based
    width: float
    height: float
    words: List[Word] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    start: int  # inclusive word index within the page
    end: int  # exclusive word index within the page
    text: str


def assign_lines(words: List[Word], y_tolerance: float = 3.0) -> List[Word]:
    """Assign a line index to each word, in the order given (pdfplumber reading order).

    A new line starts when the word's top moves more than ``y_tolerance`` away from the
    current line's top, or when the word jumps back left of the previous word (a wrap).
    """
    line = -1
    line_top = None
    prev_x1 = None
    for w in words:
        wraps_back = prev_x1 is not None and w.x0 < prev_x1 - 1.0
        moved = line_top is None or abs(w.top - line_top) > y_tolerance
        if moved or wraps_back:
            line += 1
            line_top = w.top
        w.line = line
        prev_x1 = w.x1
    return words


def extract_pages(pdf_path: str) -> List[PageText]:
    """Extract words with cropbox-relative coordinates for every page."""
    pages: List[PageText] = []
    with pdfplumber.open(pdf_path) as pdf:
        for index, page in enumerate(pdf.pages):
            off_x, off_top = page.cropbox[0], page.cropbox[1]
            width = page.cropbox[2] - page.cropbox[0]
            height = page.cropbox[3] - page.cropbox[1]
            raw = page.extract_words(x_tolerance=3, y_tolerance=3, keep_blank_chars=False)
            words = [
                Word(
                    text=w["text"],
                    x0=float(w["x0"]) - off_x,
                    top=float(w["top"]) - off_top,
                    x1=float(w["x1"]) - off_x,
                    bottom=float(w["bottom"]) - off_top,
                )
                for w in raw
            ]
            pages.append(PageText(page=index, width=width, height=height, words=assign_lines(words)))
    return pages


def span_text(words: List[Word], start: int, end: int) -> str:
    return " ".join(w.text for w in words[start:end])


def _ends_sentence(word: Word) -> bool:
    return bool(_SENTENCE_END.search(word.text))


def chunk_words(
    words: List[Word],
    doc_id: str,
    page: int,
    target_chars: int = 500,
    max_chars: int = 800,
) -> List[Chunk]:
    """Split a page's words into consecutive chunks that prefer to end at sentence boundaries.

    A chunk closes at the first sentence end once it holds at least ``target_chars``
    characters, or at the last sentence end seen if it would exceed ``max_chars``, or
    at the current word when no sentence end is available.
    """
    chunks: List[Chunk] = []
    start = 0
    n = len(words)
    while start < n:
        chars = 0
        last_sentence_end = None  # exclusive index
        end = start
        cut = None
        while end < n:
            chars += len(words[end].text) + (1 if end > start else 0)
            if chars > max_chars and end > start:
                cut = last_sentence_end if last_sentence_end is not None else end
                break
            end += 1
            if _ends_sentence(words[end - 1]):
                last_sentence_end = end
                if chars >= target_chars:
                    cut = end
                    break
        if cut is None:
            cut = end
        if cut <= start:  # a single word longer than max_chars
            cut = start + 1
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}:{page}:{start}",
                doc_id=doc_id,
                page=page,
                start=start,
                end=cut,
                text=span_text(words, start, cut),
            )
        )
        start = cut
    return chunks


def sentence_spans(words: List[Word], start: int, end: int) -> List[Tuple[int, int]]:
    """Split ``words[start:end]`` into sentence spans ``(s, e)`` on terminal punctuation."""
    spans: List[Tuple[int, int]] = []
    s = start
    for i in range(start, end):
        if _ends_sentence(words[i]):
            spans.append((s, i + 1))
            s = i + 1
    if s < end:
        spans.append((s, end))
    return spans


def line_boxes(words: List[Word], start: int, end: int) -> List[BBox]:
    """One bounding box per visual line covering exactly ``words[start:end]``."""
    boxes: List[BBox] = []
    current: List[Word] = []
    for w in words[start:end]:
        if current and w.line != current[-1].line:
            boxes.append(_merge(current))
            current = []
        current.append(w)
    if current:
        boxes.append(_merge(current))
    return boxes


def _merge(ws: List[Word]) -> BBox:
    return (
        min(w.x0 for w in ws),
        min(w.top for w in ws),
        max(w.x1 for w in ws),
        max(w.bottom for w in ws),
    )
