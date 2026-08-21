"""Verify that furniture material assignment follows structural dominance,
not visual appearance. Per paper §4.3: a sofa is wood frame + fabric +
cushion — RF sees mostly wood. Same for upholstered chairs, beds, etc."""
from __future__ import annotations
import sys
from pathlib import Path

SKILL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SKILL))

from lib.scene_gen.exporters.materials import get_material_for_category  # noqa: E402

# Categories where structural-dominance differs from visual appearance.
# Each entry: (furniture_category, expected_material, justification)
STRUCTURAL_DOMINANCE_RULES = [
    ("sofa",            "wood",     "wood frame dominates over fabric/cushion"),
    ("armchair",        "wood",     "wood frame dominates"),
    ("chair_upholstered","wood",    "wood frame dominates"),
    ("bed",             "wood",     "wood frame dominates over mattress"),
    ("bookshelf",       "wood",     "wood structure"),
    ("desk",            "wood",     "primary material"),
    ("table",           "wood",     "primary material"),
    ("filing_cabinet",  "metal",    "metal body — RF block"),
    ("refrigerator",    "metal",    "metal exterior — RF block"),
    ("monitor",         "metal",    "metal/electronics chassis"),
    ("window",          "glass",    "glass aperture"),
    ("door",            "wood",     "default door material"),
]


def test_structural_dominance_assignments():
    failures = []
    for category, expected, why in STRUCTURAL_DOMINANCE_RULES:
        actual = get_material_for_category(category)
        if actual != expected:
            failures.append(f"  {category!r}: got {actual!r}, expected {expected!r} ({why})")
    assert not failures, "structural-dominance violations:\n" + "\n".join(failures)
