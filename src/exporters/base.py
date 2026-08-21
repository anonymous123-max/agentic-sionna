"""Base protocol for exporters.

All exporters implement this protocol for consistent API across PNG, ASCII, and future formats.
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.models.room import Room


@runtime_checkable
class Exporter(Protocol):
    """Protocol for room exporters.

    Exporters convert a Room model to a specific output format.
    Uses typing.Protocol for structural subtyping (duck typing).
    """

    def export(self, room: Room) -> bytes:
        """Export room to bytes.

        Args:
            room: Room model to export

        Returns:
            Exported data as bytes
        """
        ...

    def export_file(self, room: Room, path: Path) -> None:
        """Export room to a file.

        Args:
            room: Room model to export
            path: Output file path
        """
        ...
