"""ITU radio material definitions and furniture-category mapping.

Maps 3D-FUTURE furniture categories to ITU-R P.2040-3 radio materials
for realistic wireless propagation modeling in Sionna RT.
"""

from typing import Set


# Valid ITU material names from ITU-R P.2040-3
# These are the only materials accepted by Sionna's itu-radio-material plugin
ITU_MATERIALS: Set[str] = {
    'vacuum',
    'concrete',
    'brick',
    'plasterboard',
    'wood',
    'glass',
    'ceiling_board',
    'chipboard',
    'plywood',
    'marble',
    'floorboard',
    'metal',
    'very_dry_ground',
    'medium_dry_ground',
    'wet_ground',
}

# Maps furniture category (lowercase, normalized) to ITU material name
# Based on 3D-FUTURE catalog categories and realistic material composition
CATEGORY_MATERIAL_MAP: dict[str, str] = {
    # === Beds (wood frames) ===
    "bed": "wood",
    "bed frame": "wood",
    "bunk bed": "wood",
    "double bed": "wood",
    "single bed": "wood",
    "kids bed": "wood",
    "king-size bed": "wood",
    "couch bed": "wood",

    # === Sofas (wood/metal frame with fabric - frame is RF-significant) ===
    "sofa": "wood",
    "armchair": "wood",
    "chaise longue sofa": "wood",
    "l-shaped sofa": "wood",
    "lazy sofa": "wood",
    "loveseat sofa": "wood",
    "two-seat sofa": "wood",
    "three-seat / multi-person sofa": "wood",
    "three-seat / multi-seat sofa": "wood",
    "u-shaped sofa": "wood",

    # === Tables - wood ===
    "table": "wood",
    "desk": "wood",
    "dining table": "wood",
    "dressing table": "wood",

    # === Tables - glass (coffee tables, end tables often have glass tops) ===
    "coffee table": "glass",
    "corner/side table": "glass",
    "round end table": "glass",
    "tea table": "glass",
    "bar": "glass",

    # === Chairs - wood ===
    "chair": "wood",
    "dining chair": "wood",
    "dressing chair": "wood",
    "classic chinese chair": "wood",

    # === Chairs - metal (office, folding, hanging) ===
    "barstool": "metal",
    "folding chair": "metal",
    "hanging chair": "metal",
    "lounge chair / book-chair / computer chair": "metal",
    "lounge chair / cafe chair / office chair": "metal",
    "office chair": "metal",
    "computer chair": "metal",

    # === Storage/Cabinets - wood ===
    "cabinet": "wood",
    "nightstand": "wood",
    "wardrobe": "wood",
    "bookcase / jewelry armoire": "wood",
    "bookcase": "wood",
    "bookshelf": "wood",
    "shelf": "wood",
    "tv stand": "wood",
    "shoe cabinet": "wood",
    "sideboard / side cabinet / console": "wood",
    "sideboard / side cabinet / console table": "wood",
    "drawer chest / corner cabinet": "wood",
    "children cabinet": "wood",
    "dresser": "wood",

    # === Storage - glass (wine cabinets often have glass doors) ===
    "wine cabinet": "glass",
    "wine cooler": "glass",
    "display cabinet": "glass",

    # === Lighting - metal (frames, fixtures) ===
    "lamp": "metal",
    "ceiling lamp": "metal",
    "floor lamp": "metal",
    "pendant lamp": "metal",
    "wall lamp": "metal",
    "lighting": "metal",

    # === Stools/Ottomans - wood ===
    "stool": "wood",
    "footstool / sofastool / bed end stool / stool": "wood",
    "pier": "wood",

    # === Metal furniture ===
    "filing_cabinet": "metal",
    "metal_shelf": "metal",
    "metal_chair": "metal",

    # === Explicit glass ===
    "glass_table": "glass",
    "glass table": "glass",

    # Default fallback for unknown categories
    "_default": "wood",
}


def get_material_for_category(category: str) -> str:
    """Get ITU material for furniture category.

    Uses smart matching:
    1. Exact match (case-insensitive)
    2. Partial match (category contains a key or key contains category)
    3. Keyword match (bed, sofa, chair, table, lamp, cabinet, etc.)
    4. Default fallback to wood

    Args:
        category: Furniture category from 3D-FUTURE catalog

    Returns:
        ITU material name (e.g., 'wood', 'glass', 'metal')
    """
    cat_lower = category.lower().strip()

    # 1. Exact match
    if cat_lower in CATEGORY_MATERIAL_MAP:
        return CATEGORY_MATERIAL_MAP[cat_lower]

    # 2. Check if any key is contained in category or vice versa
    for key, material in CATEGORY_MATERIAL_MAP.items():
        if key == "_default":
            continue
        if key in cat_lower or cat_lower in key:
            return material

    # 3. Keyword-based fallback for common furniture types
    keywords = {
        "bed": "wood",
        "sofa": "wood",
        "couch": "wood",
        "chair": "wood",  # Default chairs to wood, office chairs matched above
        "table": "wood",  # Default tables to wood
        "desk": "wood",
        "cabinet": "wood",
        "shelf": "wood",
        "wardrobe": "wood",
        "dresser": "wood",
        "nightstand": "wood",
        "lamp": "metal",
        "light": "metal",
        "stool": "wood",
        "ottoman": "wood",
        "bench": "wood",
        "mirror": "glass",
        "tv": "wood",  # TV stands
    }
    for keyword, material in keywords.items():
        if keyword in cat_lower:
            return material

    # 4. Default fallback
    return CATEGORY_MATERIAL_MAP["_default"]
