"""Image parser for extracting room structure from floor plan images.

Uses Claude's vision capabilities to analyze floor plan images and extract
structured room specifications with confidence levels.
"""

import base64
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from anthropic import Anthropic


class ImageExtractedRoom(BaseModel):
    """Room structure extracted from image with confidence.

    Attributes:
        width_estimate: Estimated room width in meters
        length_estimate: Estimated room length in meters
        confidence: Confidence in dimension estimates
        doors: Walls that have doors
        windows: Walls that have windows
        room_type: Inferred room type
        notes: Any caveats or uncertainties about the extraction
    """

    width_estimate: float = Field(description="Estimated room width in meters")
    length_estimate: float = Field(description="Estimated room length in meters")
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in dimension estimates - high if scale visible, low if guessing"
    )
    doors: list[Literal["north", "south", "east", "west"]] = Field(
        description="Walls that have doors (north=top, south=bottom, east=right, west=left)"
    )
    windows: list[Literal["north", "south", "east", "west"]] = Field(
        description="Walls that have windows"
    )
    room_type: str = Field(
        description="Inferred room type: bedroom, living, office, kitchen, bathroom"
    )
    notes: str = Field(
        description="Any caveats or uncertainties about the extraction"
    )


IMAGE_SYSTEM_PROMPT = """Analyze this floor plan or room image and extract the room structure.

IMPORTANT: Your spatial measurements may be imprecise. Always set confidence appropriately:
- "high": Clear scale reference visible (measurements marked, grid, known object size)
- "medium": Some reference points but no explicit scale
- "low": No scale reference, purely estimating from image proportions

Identify door and window locations by wall (north=top of image, south=bottom, east=right, west=left).

If you cannot determine dimensions with any confidence, use reasonable defaults:
- bedroom: 4m x 5m
- living room: 5m x 6m
- office: 3m x 4m
- kitchen: 3m x 4m
- bathroom: 2m x 3m

Call the extract_room_structure tool with your analysis."""


class ImageParser:
    """Parse room images into structured RoomSpec.

    Uses Claude's vision capabilities to analyze floor plan images
    and extract room dimensions, door/window locations, and room type.

    Example:
        parser = ImageParser()
        extracted = parser.parse_image("floor_plan.png")
        if parser.needs_confirmation(extracted):
            print("Low confidence - please verify dimensions")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250514",
    ):
        """Initialize parser with API client.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use for vision analysis
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def parse_image(self, image_path: str | Path) -> ImageExtractedRoom:
        """Extract room structure from floor plan image.

        Args:
            image_path: Path to image file (JPG, PNG, WebP)

        Returns:
            ImageExtractedRoom with dimensions, doors, windows, and confidence
        """
        image_path = Path(image_path)
        image_data = base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")

        # Determine media type from extension
        suffix = image_path.suffix.lower()
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
        }
        media_type = media_types.get(suffix, "image/jpeg")

        # Define tool for structured extraction (same pattern as RoomParser)
        tools = [
            {
                "name": "extract_room_structure",
                "description": "Extract room structure from the analyzed floor plan image",
                "input_schema": ImageExtractedRoom.model_json_schema(),
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=IMAGE_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "tool", "name": "extract_room_structure"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract the room structure from this floor plan image.",
                        },
                    ],
                }
            ],
        )

        # Extract the tool use content
        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_room_structure":
                return ImageExtractedRoom.model_validate(block.input)

        # Fallback: should not reach here with tool_choice
        raise ValueError("Failed to extract room structure from image")

    def needs_confirmation(self, extracted: ImageExtractedRoom) -> bool:
        """Check if extraction needs user confirmation.

        Returns True for low/medium confidence or if notes indicate uncertainty.

        Args:
            extracted: ImageExtractedRoom from parse_image()

        Returns:
            True if user should verify the extraction
        """
        return extracted.confidence in ("low", "medium")
