from typing import Protocol


class HealthProtocol(Protocol):
    async def is_healthy(self) -> bool: ...
