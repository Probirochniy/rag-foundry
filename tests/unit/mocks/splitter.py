from src.core.protocols.splitter import TextSplitterProtocol


class TextSplitterMock(TextSplitterProtocol):
    def split(self, text: str) -> list[str]:
        return ["chunk1", "chunk2", "chunk3"]
