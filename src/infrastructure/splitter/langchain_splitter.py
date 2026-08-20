from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.protocols.splitter import TextSplitterProtocol


class LangChainSplitterAdapter(TextSplitterProtocol):
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
            keep_separator=True,
        )

    def split(self, text: str) -> list[str]:
        cleaned = text.strip()
        if not cleaned:
            return []
        return self._splitter.split_text(cleaned)
