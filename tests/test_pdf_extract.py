from pdf_extract import (
    Word,
    assign_lines,
    chunk_words,
    extract_pages,
    line_boxes,
    sentence_spans,
    span_text,
)


def _w(text, x0, top, width=10.0, height=10.0):
    return Word(text=text, x0=x0, top=top, x1=x0 + width, bottom=top + height)


def test_assign_lines_groups_by_vertical_position():
    words = [_w("a", 0, 100), _w("b", 20, 101), _w("c", 0, 120), _w("d", 20, 120)]
    out = assign_lines(words)
    assert [w.line for w in out] == [0, 0, 1, 1]


def test_assign_lines_starts_new_line_when_x_wraps_back_within_tolerance():
    # Words on one visual line, even far apart (two columns), stay on one line.
    words = [_w("left1", 0, 100), _w("left2", 20, 100), _w("right1", 300, 100)]
    out = assign_lines(words)
    assert out[0].line == out[1].line == out[2].line
    # A word that jumps back left within the y tolerance starts a new line.
    words = [_w("a", 0, 100), _w("b", 20, 100), _w("c", 0, 102)]
    out = assign_lines(words)
    assert out[2].line == 1


def test_extract_pages_returns_words_with_lines_and_page_size(sample_pdf):
    pages = extract_pages(sample_pdf)
    assert len(pages) == 2
    p0 = pages[0]
    assert p0.page == 0
    assert 600 < p0.width < 620 and 780 < p0.height < 800  # LETTER = 612x792
    texts = [w.text for w in p0.words]
    assert texts[:3] == ["Employee", "Benefits", "Overview."]
    assert all(w.line >= 0 for w in p0.words)
    assert p0.words[0].line == 0 and p0.words[3].line == 1
    assert all(0 <= w.x0 < w.x1 <= p0.width for w in p0.words)
    assert all(0 <= w.top < w.bottom <= p0.height for w in p0.words)


def test_chunk_words_covers_all_words_in_order_without_gaps():
    words = assign_lines([_w(f"w{i}.", (i % 10) * 20, (i // 10) * 15) for i in range(60)])
    chunks = chunk_words(words, doc_id="d", page=0, target_chars=60, max_chars=100)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(words)
    for a, b in zip(chunks, chunks[1:]):
        assert a.end == b.start
    assert all(len(c.text) <= 100 for c in chunks)
    assert all(c.text == span_text(words, c.start, c.end) for c in chunks)
    assert chunks[0].chunk_id == "d:0:0" and chunks[0].doc_id == "d" and chunks[0].page == 0


def test_chunk_words_prefers_sentence_boundaries():
    texts = "one two three four. five six seven eight nine ten. eleven twelve".split()
    words = assign_lines([_w(t, i * 30, 0) for i, t in enumerate(texts)])
    # "one two three four." is 19 chars, so a 15-char target closes at that sentence end.
    chunks = chunk_words(words, doc_id="d", page=0, target_chars=15, max_chars=60)
    assert chunks[0].text == "one two three four."
    assert chunks[1].text == "five six seven eight nine ten."
    assert chunks[2].text == "eleven twelve"


def test_chunk_words_empty_page_returns_no_chunks():
    assert chunk_words([], doc_id="d", page=3) == []


def test_sentence_spans_splits_on_terminal_punctuation():
    texts = 'Hello there. Second "one?" Third (yes!) last words'.split()
    words = [_w(t, i * 30, 0) for i, t in enumerate(texts)]
    spans = sentence_spans(words, 0, len(words))
    assert spans == [(0, 2), (2, 4), (4, 6), (6, 8)]
    assert span_text(words, *spans[1]) == 'Second "one?"'


def test_line_boxes_one_box_per_line_covering_only_the_span():
    words = assign_lines(
        [
            _w("a", 0, 100),
            _w("b", 20, 100),
            _w("c", 40, 100),
            _w("d", 0, 120),
            _w("e", 20, 120),
        ]
    )
    boxes = line_boxes(words, 1, 4)  # b, c, d
    assert boxes == [(20.0, 100.0, 50.0, 110.0), (0.0, 120.0, 10.0, 130.0)]
