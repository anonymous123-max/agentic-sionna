"""ASCII floor plan exporter.

Renders room layouts as ASCII art for terminal display.
Supports furniture visualization with category-specific characters.
"""

from pathlib import Path
from typing import Dict, List, Optional

from shapely import Point, Polygon

from src.models.room import Room
from src.utils.geometry import corners


# Character mapping for furniture categories
CATEGORY_CHARS: Dict[str, str] = {
    "bed": "B",
    "sofa": "S",
    "table": "T",
    "chair": "C",
    "desk": "D",
    "wardrobe": "W",
    "nightstand": "N",
    "bookshelf": "K",
    "cabinet": "A",
    "lamp": "L",
}

DEFAULT_CHAR = "#"


class ASCIIExporter:
    """Export room layouts as ASCII art.

    Creates terminal-friendly visualizations with:
    - Room boundary using +, -, | characters
    - Door openings shown as gaps in walls
    - Furniture items as category-specific characters

    Coordinate system note:
    - Room coordinates: origin at SW corner, Y increases north
    - Terminal grid: row 0 at top, row increases downward
    - Transform: grid_row = (rows - 1) - int(y * chars_per_meter)
    """

    def __init__(self, chars_per_meter: float = 2.0):
        """Initialize ASCII exporter.

        Args:
            chars_per_meter: Grid resolution (default 2.0 chars per meter)
        """
        self.chars_per_meter = chars_per_meter

    def export(self, room: Room) -> bytes:
        """Export room to UTF-8 encoded ASCII bytes.

        Args:
            room: Room model to export

        Returns:
            ASCII art as UTF-8 encoded bytes
        """
        return self.export_str(room).encode("utf-8")

    def export_str(self, room: Room) -> str:
        """Export room to ASCII string.

        Args:
            room: Room model to export

        Returns:
            ASCII art string
        """
        # Calculate grid dimensions
        cols = int(room.width * self.chars_per_meter) + 1
        rows = int(room.length * self.chars_per_meter) + 1

        # Initialize grid with spaces
        grid: List[List[str]] = [[" " for _ in range(cols)] for _ in range(rows)]

        # Draw room boundary
        self._draw_boundary(grid, room, rows, cols)

        # Draw doors as gaps
        self._draw_doors(grid, room, rows, cols)

        # Draw furniture
        self._draw_furniture(grid, room, rows, cols)

        # Join rows into string
        return "\n".join("".join(row) for row in grid)

    def export_file(self, room: Room, path: Path) -> None:
        """Export room to a text file.

        Args:
            room: Room model to export
            path: Output file path
        """
        text = self.export_str(room)
        path.write_text(text, encoding="utf-8")

    def _draw_boundary(
        self, grid: List[List[str]], room: Room, rows: int, cols: int
    ) -> None:
        """Draw room boundary on grid.

        Uses:
        - '+' for corners
        - '-' for horizontal walls (top/bottom)
        - '|' for vertical walls (left/right)
        """
        # Top wall (north - row 0 in terminal)
        for c in range(cols):
            grid[0][c] = "-"

        # Bottom wall (south - last row in terminal)
        for c in range(cols):
            grid[rows - 1][c] = "-"

        # Left wall (west)
        for r in range(rows):
            grid[r][0] = "|"

        # Right wall (east)
        for r in range(rows):
            grid[r][cols - 1] = "|"

        # Corners
        grid[0][0] = "+"
        grid[0][cols - 1] = "+"
        grid[rows - 1][0] = "+"
        grid[rows - 1][cols - 1] = "+"

    def _draw_doors(
        self, grid: List[List[str]], room: Room, rows: int, cols: int
    ) -> None:
        """Draw doors as gaps in walls."""
        for door in room.doors:
            start = int(door.position * self.chars_per_meter)
            end = int((door.position + door.width) * self.chars_per_meter)

            if door.wall == "south":
                # South wall is bottom in terminal (last row)
                for c in range(max(1, start), min(cols - 1, end + 1)):
                    grid[rows - 1][c] = " "
            elif door.wall == "north":
                # North wall is top in terminal (row 0)
                for c in range(max(1, start), min(cols - 1, end + 1)):
                    grid[0][c] = " "
            elif door.wall == "west":
                # West wall is left column
                # Y position in room -> row in grid (inverted)
                for pos in range(max(1, start), min(rows - 1, end + 1)):
                    grid_row = (rows - 1) - pos
                    if 0 < grid_row < rows - 1:
                        grid[grid_row][0] = " "
            elif door.wall == "east":
                # East wall is right column
                for pos in range(max(1, start), min(rows - 1, end + 1)):
                    grid_row = (rows - 1) - pos
                    if 0 < grid_row < rows - 1:
                        grid[grid_row][cols - 1] = " "

    def _draw_furniture(
        self, grid: List[List[str]], room: Room, rows: int, cols: int
    ) -> None:
        """Draw furniture items on grid.

        Uses geometry.corners() to get rotated furniture footprint,
        then fills grid cells that fall within the polygon.
        """
        for item in room.furniture:
            char = CATEGORY_CHARS.get(item.category, DEFAULT_CHAR)

            # Get furniture polygon corners
            pts = corners(
                item.position.x,
                item.position.y,
                item.position.theta,
                item.dimensions.width,
                item.dimensions.depth,
            )
            poly = Polygon(pts)

            # Fill grid cells within the polygon
            # Skip boundary cells (r=0, r=rows-1, c=0, c=cols-1)
            for r in range(1, rows - 1):
                for c in range(1, cols - 1):
                    # Convert grid position to room coordinates
                    x = c / self.chars_per_meter
                    # Invert Y: terminal row 0 = max Y (north)
                    y = (rows - 1 - r) / self.chars_per_meter

                    if poly.contains(Point(x, y)):
                        grid[r][c] = char
