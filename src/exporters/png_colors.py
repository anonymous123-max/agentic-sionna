"""Color palettes for PNG floor plan rendering.

Defines color constants for furniture categories, room elements,
and outdoor scene elements used by the PNGExporter.
"""

from typing import Dict


# Color palette for furniture categories (RGB 0-1)
CATEGORY_COLORS: Dict[str, tuple] = {
    "bed": (0.545, 0.271, 0.075, 0.9),       # saddle brown
    "sofa": (0.255, 0.412, 0.882, 0.9),      # royal blue
    "chair": (0.133, 0.545, 0.133, 0.9),     # forest green
    "table": (0.855, 0.647, 0.125, 0.9),     # goldenrod
    "desk": (0.545, 0.0, 0.545, 0.9),        # dark magenta
    "wardrobe": (0.184, 0.310, 0.310, 0.9),  # dark slate gray
    "nightstand": (0.824, 0.412, 0.118, 0.9),  # chocolate
    "bookshelf": (0.333, 0.420, 0.184, 0.9),   # dark olive green
    "cabinet": (0.627, 0.322, 0.176, 0.9),     # sienna
    "lamp": (1.0, 0.843, 0.0, 0.9),            # gold
}

DEFAULT_COLOR = (0.5, 0.5, 0.5, 0.9)  # gray for unknown categories

# Room colors
FLOOR_COLOR = (0.96, 0.96, 0.94, 1.0)  # off-white
WALL_COLOR = (0.78, 0.78, 0.78, 1.0)   # light gray
WINDOW_COLOR = (0.529, 0.808, 0.922, 0.8)  # light sky blue, 80% opacity
WINDOW_EDGE_COLOR = (0.2, 0.5, 0.7)  # darker blue for visibility

# Outdoor element colors
GROUND_COLORS: Dict[str, tuple] = {
    "wet_ground": (0.56, 0.93, 0.56, 1.0),      # lightgreen
    "medium_dry_ground": (0.56, 0.93, 0.56, 1.0),  # lightgreen
    "concrete": (0.83, 0.83, 0.83, 1.0),        # lightgray
    "very_dry_ground": (0.96, 0.64, 0.38, 1.0),  # sandybrown
}
BUILDING_COLOR = (0.7, 0.7, 0.7, 1.0)  # concrete gray
ROAD_COLOR = (0.3, 0.3, 0.3, 1.0)  # dark gray
TREE_COLORS: Dict[str, tuple] = {
    "deciduous": (0.13, 0.55, 0.13, 0.9),  # forest green
    "conifer": (0.0, 0.39, 0.0, 0.9),      # dark green
}
