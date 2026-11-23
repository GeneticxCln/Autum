from autum.text import SentenceChunker, speakable


def stream(text, chunker=None, step=3):
    chunker = chunker or SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += chunker.feed(text[i : i + step])
    return out + chunker.flush()


def test_splits_at_sentence_ends_but_not_decimals_or_abbreviations():
    text = "It costs 3.50 today. Dr. Lee agrees, e.g. on Mondays! Really? Yes."
    assert stream(text, SentenceChunker(first_clause_chars=1000)) == [
        "It costs 3.50 today.",
        "Dr. Lee agrees, e.g. on Mondays!",
        "Really?",
        "Yes.",
    ]


def test_first_chunk_may_end_at_a_clause_so_audio_starts_sooner():
    text = "Well, after thinking about this question carefully, the answer is yes. Later, more."
    chunks = stream(text, SentenceChunker(first_clause_chars=40))
    assert chunks[0] == "Well, after thinking about this question carefully,"
    assert chunks[1:] == ["the answer is yes.", "Later, more."]


def test_newlines_end_chunks_and_blank_lines_are_skipped():
    assert stream("One\n\n\nTwo") == ["One", "Two"]


def test_speakable_removes_markup_urls_and_reasoning():
    raw = "<think>hidden plan</think>**Sure** — see https://example.com for `details`."
    assert speakable(raw) == "Sure — see for details."


def test_chunker_and_speakable_strip_gemma4_and_lfm3_thought_blocks():
    gemma_raw = "<|channel>thought\nThinking about user greeting.\n<channel|>Hello there. How can I help?"
    lfm_raw = "<|think|>\nInternal reasoning\n<|/think|>Hi! Ready when you are."
    assert stream(gemma_raw, step=2) == ["Hello there.", "How can I help?"]
    assert stream(lfm_raw, step=2) == ["Hi!", "Ready when you are."]
    assert speakable(gemma_raw) == "Hello there. How can I help?"
    assert speakable(lfm_raw) == "Hi! Ready when you are."


def test_text_without_punctuation_is_cut_at_a_space_after_max_chars():
    chunks = stream("word " * 30, SentenceChunker(max_chars=50))
    assert all(len(c) <= 50 for c in chunks)
    assert " ".join(chunks).split() == ["word"] * 30
