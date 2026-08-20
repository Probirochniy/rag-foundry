from src.infrastructure.splitter.langchain_splitter import LangChainSplitterAdapter


def test_langchain_splitter_splits_text() -> None:
    splitter = LangChainSplitterAdapter(chunk_size=10, chunk_overlap=2)
    text = "text\n\nbottom text"
    chunks = splitter.split(text)

    assert len(chunks) > 1
    assert all(len(c) <= 25 for c in chunks)
    assert splitter.split("") == []
