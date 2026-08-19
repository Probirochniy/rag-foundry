from typing import Protocol


class TextSplitterProtocol(Protocol):
    def split(self, text: str) -> list[str]: ...
