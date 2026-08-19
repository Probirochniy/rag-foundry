from src.core.protocols.splitter import TextSplitterProtocol


class SlidingWindowSplitter(TextSplitterProtocol):
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        chunks: list[str] = []
        start = 0
        text_len = len(text)
        step = max(1, self._chunk_size - self._chunk_overlap)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_len:
                break
            start += step

        return chunks
