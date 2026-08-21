"""3D-FUTURE furniture catalog interface.

Provides access to the 3D-FUTURE dataset for furniture model selection and
dimension extraction. The catalog loads model_info.json and indexes models
by category for efficient searching.

Usage:
    catalog = FutureCatalog("/path/to/3D-FUTURE-model")
    sofas = catalog.get_by_category("sofa")
    dims = catalog.get_dimensions(sofas[0]["model_id"])
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh

from src.models.room import BoundingBox


# Category-based default facing directions.
# 3D-FUTURE models are rendered with consistent camera angles per category.
# Visual inspection of 8+ models per category confirms these directions.
CATEGORY_DEFAULT_FACING: Dict[str, str] = {
    "sofa": "+Z",           # All sofas face +Z (toward camera)
    "wardrobe": "-Z",       # All wardrobes face -Z (back to camera)
    "desk": "+Z",           # All desks face +Z
    "nightstand": "-Z",     # All nightstands face -Z
    "bookcase": "-Z",       # All bookcases face -Z
    "bed": "+Z",            # All beds face +Z (foot toward camera)
    "chair": "+Z",          # All chairs face +Z (seat toward camera)
    "armchair": "+Z",       # All armchairs face +Z
    "table": "+Z",          # Tables face +Z (default)
    "cabinet": "-Z",        # Cabinets face -Z (like wardrobes)
    "lamp": "+Z",           # Lamps face +Z (default)
}

# Rotation offset to make model face +Z (standard forward direction)
FACING_TO_OFFSET: Dict[str, float] = {
    "+Z": 0.0,
    "-Z": math.pi,
    "+X": -math.pi / 2,
    "-X": math.pi / 2,
}


class FutureCatalog:
    """Interface to 3D-FUTURE furniture dataset.

    The 3D-FUTURE dataset contains 12,400+ furniture models with metadata
    including category, style, theme, and material. Model dimensions are
    not stored in the metadata and must be computed from OBJ bounding boxes.

    Attributes:
        path: Path to the 3D-FUTURE-model directory
    """

    def __init__(self, dataset_path: Path | str) -> None:
        """Load model_info.json and build category index.

        Args:
            dataset_path: Path to 3D-FUTURE-model directory
                         (contains model_info.json and model subdirectories)
        """
        self.path = Path(dataset_path)
        self._models: Dict[str, dict] = {}
        self._category_index: Dict[str, List[str]] = {}
        self._dimension_cache: Dict[str, BoundingBox] = {}
        self._orientation_cache: Dict[str, float] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Parse model_info.json into searchable index.

        model_info.json is a list of dicts with keys:
        model_id, super-category, category, style, theme, material

        Builds:
        - _models: dict keyed by model_id
        - _category_index: dict mapping lowercase category to list of model_ids
        """
        info_path = self.path / "model_info.json"
        with open(info_path) as f:
            models_list = json.load(f)

        for model in models_list:
            model_id = model["model_id"]
            self._models[model_id] = model

            # Build category index with lowercase keys
            # Handle None category values in dataset
            raw_category = model.get("category")
            category = (raw_category or "").lower()
            if category:
                if category not in self._category_index:
                    self._category_index[category] = []
                self._category_index[category].append(model_id)

    def get_by_category(
        self, category: str, style: Optional[str] = None
    ) -> List[dict]:
        """Find models matching category (case-insensitive substring match).

        Args:
            category: Category to search for (e.g., "sofa", "bed", "table")
            style: Optional style filter (e.g., "Modern", "Minimalist")

        Returns:
            List of model info dicts matching the criteria
        """
        category_lower = category.lower()
        matches = []

        # Substring match across all categories
        for cat_name, model_ids in self._category_index.items():
            if category_lower in cat_name:
                for model_id in model_ids:
                    model = self._models[model_id]
                    # Apply style filter if provided
                    if style is None:
                        matches.append(model)
                    else:
                        model_style = model.get("style") or ""
                        if style.lower() in model_style.lower():
                            matches.append(model)

        return matches

    def get_model_path(self, model_id: str) -> Path:
        """Return path to model directory (contains raw_model.obj).

        Args:
            model_id: The model UUID from model_info.json

        Returns:
            Path to the model directory
        """
        return self.path / model_id

    def get_dimensions(self, model_id: str) -> BoundingBox:
        """Load OBJ and compute bounding box dimensions.

        Dimensions are cached after first load for efficiency.

        Uses trimesh to load raw_model.obj and compute the bounding box.
        Returns BoundingBox(width=x_extent, depth=z_extent, height=y_extent).

        Note: 3D-FUTURE models use Y-up convention (standard for 3D modeling
        software like Blender, Maya). When loaded into Sionna (Z-up), models
        need rotation around X-axis.

        Args:
            model_id: The model UUID from model_info.json

        Returns:
            BoundingBox with width, depth, and height in meters
        """
        if model_id not in self._dimension_cache:
            obj_path = self.path / model_id / "raw_model.obj"
            mesh = trimesh.load(obj_path, force="mesh")
            # mesh.bounds is [[min_x, min_y, min_z], [max_x, max_y, max_z]]
            bounds = mesh.bounds
            extents = bounds[1] - bounds[0]

            # 3D-FUTURE uses Y-up, so:
            # X extent = width (perpendicular to facing)
            # Y extent = height (vertical in Y-up)
            # Z extent = depth (forward/back in Y-up)
            self._dimension_cache[model_id] = BoundingBox(
                width=float(extents[0]),
                depth=float(extents[2]),  # Z in Y-up = depth
                height=float(extents[1]),  # Y in Y-up = height
            )

        return self._dimension_cache[model_id]

    def get_random_model(
        self,
        category: str,
        style: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Get random model matching category/style.

        Uses provided RNG for reproducibility (INFRA-02 requirement).

        Args:
            category: Category to search for
            style: Optional style filter
            rng: NumPy random generator for reproducibility

        Returns:
            Single model info dict

        Raises:
            ValueError: If no models match the criteria
        """
        matches = self.get_by_category(category, style)
        if not matches:
            raise ValueError(
                f"No models found for category='{category}', style='{style}'"
            )

        if rng is None:
            rng = np.random.default_rng()

        idx = rng.integers(0, len(matches))
        return matches[idx]

    def get_orientation_offset(self, model_id: str, category: str) -> float:
        """Get rotation offset to normalize model to face +Z.

        Uses category-based defaults from CATEGORY_DEFAULT_FACING.
        Falls back to +Z (no offset) for unknown categories.

        Args:
            model_id: The model UUID from model_info.json
            category: Furniture category (case-insensitive)

        Returns:
            Rotation offset in radians to apply when rendering
        """
        if model_id in self._orientation_cache:
            return self._orientation_cache[model_id]

        category_lower = category.lower()

        # Check for exact category match first
        if category_lower in CATEGORY_DEFAULT_FACING:
            facing = CATEGORY_DEFAULT_FACING[category_lower]
        else:
            # Check for substring match (e.g., "office chair" contains "chair")
            facing = "+Z"  # Default
            for cat_key in CATEGORY_DEFAULT_FACING:
                if cat_key in category_lower:
                    facing = CATEGORY_DEFAULT_FACING[cat_key]
                    break

        offset = FACING_TO_OFFSET.get(facing, 0.0)
        self._orientation_cache[model_id] = offset
        return offset

    @property
    def models(self) -> Dict[str, dict]:
        """Access all models indexed by model_id."""
        return self._models

    def __len__(self) -> int:
        """Return total number of models in catalog."""
        return len(self._models)
