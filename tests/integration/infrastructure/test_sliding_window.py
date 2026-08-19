from src.infrastructure.splitter.sliding_window import SlidingWindowSplitter


def test_sliding_window_splitter_splits_text_with_overlap() -> None:
    splitter = SlidingWindowSplitter(chunk_size=5, chunk_overlap=2)

    chunks = splitter.split("01234567890")

    assert chunks == ["01234", "34567", "67890"]
