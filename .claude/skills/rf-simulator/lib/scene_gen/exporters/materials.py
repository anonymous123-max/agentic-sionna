"""ITU radio material definitions and furniture-category mapping.

Maps 3D-FUTURE furniture categories to ITU-R P.2040-3 radio materials
for realistic wireless propagation modeling in Sionna RT.
"""
from __future__ import annotations

from typing import Set

ITU_MATERIALS: Set[str] = {
    "vacuum",
    "concrete",
    "brick",
    "plasterboard",
    "wood",
    "glass",
    "ceiling_board",
    "chipboard",
    "plywood",
    "marble",
    "floorboard",
    "metal",
    "very_dry_ground",
    "medium_dry_ground",
    "wet_ground",
}

# Category-to-material assignments follow STRUCTURAL DOMINANCE (paper §4.3):
# pick the constituent whose volume × conductivity dominates RF propagation,
# not the visually-dominant surface. A sofa is wood-frame-dominant; an
# upholstered chair is wood-dominant; a filing cabinet is metal-dominant.
# When in doubt: which material would block a 3.5 GHz signal more?
# See lib/scene_gen/tests/test_materials_dominance.py for the regression
# gate covering the most common categories.
CATEGORY_MATERIAL_MAP: dict[str, str] = {
    # Beds
    "bed": "wood", "bed frame": "wood", "bunk bed": "wood",
    "double bed": "wood", "single bed": "wood", "kids bed": "wood",
    "king-size bed": "wood", "couch bed": "wood",
    # Sofas
    "sofa": "wood", "armchair": "wood", "chaise longue sofa": "wood",
    "l-shaped sofa": "wood", "lazy sofa": "wood", "loveseat sofa": "wood",
    "two-seat sofa": "wood",
    "three-seat / multi-person sofa": "wood",
    "three-seat / multi-seat sofa": "wood",
    "u-shaped sofa": "wood",
    # Tables — wood
    "table": "wood", "desk": "wood", "dining table": "wood",
    "dressing table": "wood",
    # Tables — glass
    "coffee table": "glass", "corner/side table": "glass",
    "round end table": "glass", "tea table": "glass", "bar": "glass",
    # Chairs — wood
    "chair": "wood", "dining chair": "wood",
    "dressing chair": "wood", "classic chinese chair": "wood",
    # Chairs — metal
    "barstool": "metal", "folding chair": "metal", "hanging chair": "metal",
    "lounge chair / book-chair / computer chair": "metal",
    "lounge chair / cafe chair / office chair": "metal",
    "office chair": "metal", "computer chair": "metal",
    # Storage — wood
    "cabinet": "wood", "nightstand": "wood", "wardrobe": "wood",
    "bookcase / jewelry armoire": "wood", "bookcase": "wood",
    "bookshelf": "wood", "shelf": "wood", "tv stand": "wood",
    "shoe cabinet": "wood",
    "sideboard / side cabinet / console": "wood",
    "sideboard / side cabinet / console table": "wood",
    "drawer chest / corner cabinet": "wood",
    "children cabinet": "wood", "dresser": "wood",
    # Storage — glass
    "wine cabinet": "glass", "wine cooler": "glass",
    "display cabinet": "glass",
    # Lighting
    "lamp": "metal", "ceiling lamp": "metal", "floor lamp": "metal",
    "pendant lamp": "metal", "wall lamp": "metal", "lighting": "metal",
    # Stools/ottomans
    "stool": "wood", "stool_seating": "wood", "ottoman": "wood",
    "footstool / sofastool / bed end stool / stool": "wood",
    "pier": "wood", "bean_bag_chair": "wood",
    # ABO
    "headboard": "wood", "bed_frame": "wood",
    "light_fixture": "metal", "home_mirror": "glass", "bench": "wood",
    # Metal — structural dominance: metal body blocks RF
    "filing_cabinet": "metal", "metal_shelf": "metal", "metal_chair": "metal",
    "refrigerator": "metal", "fridge": "metal",
    "monitor": "metal", "tv": "metal", "television": "metal",
    # Glass apertures
    "window": "glass", "glass_table": "glass", "glass table": "glass",
    # Default
    "_default": "wood",
}


def get_material_for_category(category: str) -> str:
    """Resolve ITU material from a 3D-FUTURE category string."""
    cat_lower = category.lower().strip()
    if cat_lower in CATEGORY_MATERIAL_MAP:
        return CATEGORY_MATERIAL_MAP[cat_lower]
    for key, material in CATEGORY_MATERIAL_MAP.items():
        if key == "_default":
            continue
        if key in cat_lower or cat_lower in key:
            return material
    keywords = {
        "bed": "wood", "sofa": "wood", "couch": "wood", "chair": "wood",
        "table": "wood", "desk": "wood", "cabinet": "wood", "shelf": "wood",
        "wardrobe": "wood", "dresser": "wood", "nightstand": "wood",
        "lamp": "metal", "light": "metal", "stool": "wood",
        "ottoman": "wood", "bench": "wood", "mirror": "glass",
        "window": "glass", "tv": "metal", "monitor": "metal",
        "fridge": "metal", "refrigerator": "metal",
    }
    for keyword, material in keywords.items():
        if keyword in cat_lower:
            return material
    return CATEGORY_MATERIAL_MAP["_default"]


def resolve_material(material: str) -> str:
    """Return the input material name if it's a valid ITU material, else default."""
    if material in ITU_MATERIALS:
        return material
    # Strip ITU prefix if present
    stripped = material.replace("itu_", "").lower()
    if stripped in ITU_MATERIALS:
        return stripped
    return "wood"
