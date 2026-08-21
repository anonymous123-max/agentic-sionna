"""IRC building code constants and validation for windows.

Provides constants from the International Residential Code (IRC) for window
placement, particularly egress requirements for bedrooms and glazing ratios
for habitable rooms.

All model dimensions are in meters; these constants are in imperial (sqft/inches)
for compliance checking, with conversion done in validation functions.
"""

from typing import List

from src.models.room import Window


# Conversion constants
M2_TO_SQFT = 10.764  # square meters to square feet
M_TO_IN = 39.37      # meters to inches


# IRC Egress Requirements (2021 IRC Section R310)
# Minimum net clear opening for emergency escape and rescue
EGRESS_MIN_OPENING_AREA_SQFT = 5.7  # Minimum opening area in square feet
EGRESS_MIN_OPENING_HEIGHT_IN = 24   # Minimum opening height in inches
EGRESS_MIN_OPENING_WIDTH_IN = 20    # Minimum opening width in inches
EGRESS_MAX_SILL_HEIGHT_IN = 44      # Maximum sill height from floor in inches

# Grade floor exception (not used in v1.1, but documented)
EGRESS_GRADE_FLOOR_MIN_AREA_SQFT = 5.0

# IRC Glazing Requirements (2021 IRC Section R303.1)
# Natural light requirements for habitable rooms
MIN_GLAZING_RATIO = 0.08  # 8% of floor area minimum
MAX_GLAZING_RATIO = 0.10  # 10% of floor area typical maximum

# Ventilation requirement (not implemented in v1.1)
MIN_VENTILATION_RATIO = 0.04  # 4% openable area


def validate_egress_window(window: Window) -> List[str]:
    """Validate window meets IRC R310 egress requirements.

    Checks that a window meets the minimum requirements for emergency
    escape and rescue openings as specified in the International
    Residential Code.

    Args:
        window: Window to validate

    Returns:
        List of violation messages. Empty list if window is compliant.

    Example:
        >>> window = Window(wall="south", position=1.0, width=1.0, height=0.7, sill_height=0.6)
        >>> violations = validate_egress_window(window)
        >>> if not violations:
        ...     print("Window meets egress requirements")
    """
    violations = []

    # Convert from meters to imperial for IRC compliance check
    opening_area_sqft = (window.width * window.height) * M2_TO_SQFT
    opening_width_in = window.width * M_TO_IN
    opening_height_in = window.height * M_TO_IN
    sill_height_in = window.sill_height * M_TO_IN

    # Check minimum opening area (5.7 sqft)
    if opening_area_sqft < EGRESS_MIN_OPENING_AREA_SQFT:
        violations.append(
            f"Opening area {opening_area_sqft:.1f} sqft < "
            f"{EGRESS_MIN_OPENING_AREA_SQFT} sqft required"
        )

    # Check minimum opening width (20")
    if opening_width_in < EGRESS_MIN_OPENING_WIDTH_IN:
        violations.append(
            f"Opening width {opening_width_in:.1f}\" < "
            f"{EGRESS_MIN_OPENING_WIDTH_IN}\" required"
        )

    # Check minimum opening height (24")
    if opening_height_in < EGRESS_MIN_OPENING_HEIGHT_IN:
        violations.append(
            f"Opening height {opening_height_in:.1f}\" < "
            f"{EGRESS_MIN_OPENING_HEIGHT_IN}\" required"
        )

    # Check maximum sill height (44")
    if sill_height_in > EGRESS_MAX_SILL_HEIGHT_IN:
        violations.append(
            f"Sill height {sill_height_in:.1f}\" > "
            f"{EGRESS_MAX_SILL_HEIGHT_IN}\" maximum"
        )

    return violations
