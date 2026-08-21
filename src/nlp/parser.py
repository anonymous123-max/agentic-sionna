"""Room parser for natural language descriptions.

Converts natural language room descriptions into structured Room objects
with furniture placed via the layout optimizer.
"""

from typing import List, Optional

import numpy as np
from anthropic import Anthropic

from src.catalog.furniture import FutureCatalog
from src.models.room import BoundingBox, Door, Room
from src.nlp.prompts import CATEGORY_ALIASES, ROOM_PARSING_SYSTEM_PROMPT
from src.nlp.schemas import ParsedRoomDescription, RawFurnitureRequest
from src.optimizer.layout import FurnitureSpec, optimize_layout
from src.windows import WindowPlacer


class RoomParser:
    """Parse natural language room descriptions into Room objects.

    Uses Claude API with structured output to extract room specifications,
    then selects furniture from the 3D-FUTURE catalog and optimizes layout.

    Example:
        from src.config import SkillConfig
        from src.catalog import FutureCatalog

        config = SkillConfig()
        catalog = FutureCatalog(config.future_dataset_path)
        parser = RoomParser(catalog)
        room = parser.parse(
            "A 4 meter by 5 meter bedroom with a door on the south wall. "
            "Include a queen bed and two nightstands.",
            seed=42
        )
    """

    def __init__(
        self,
        catalog: FutureCatalog,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250514",
    ) -> None:
        """Initialize parser with catalog and API client.

        Args:
            catalog: 3D-FUTURE furniture catalog
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use for parsing
        """
        self.catalog = catalog
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def parse(self, description: str, seed: int = 42) -> Room:
        """Parse natural language description and return optimized Room.

        Args:
            description: Natural language room description
            seed: Random seed for reproducibility

        Returns:
            Room object with optimized furniture layout
        """
        # 1. Extract structured data via Claude
        raw = self._extract_structure(description)

        # 2. Map categories and select models from catalog
        furniture_specs = self._select_furniture(raw.furniture, seed)

        # 3. Build Door objects from parsed data
        doors = [
            Door(wall=d.wall, position=d.position_along_wall, width=d.width)
            for d in raw.doors
        ]

        # 4. Optimize layout
        room = optimize_layout(
            room_width=raw.room.width_meters,
            room_length=raw.room.length_meters,
            doors=doors,
            furniture_specs=furniture_specs,
            seed=seed,
        )

        # 5. Place windows based on room type
        room = self._place_windows(room, raw.room.room_type)
        return room

    def _extract_structure(self, description: str) -> ParsedRoomDescription:
        """Call Claude API with structured output.

        Uses the messages API with tool_use to extract structured data.

        Args:
            description: Natural language room description

        Returns:
            ParsedRoomDescription with extracted room data
        """
        # Define the schema as a tool for structured extraction
        tools = [
            {
                "name": "parse_room",
                "description": "Parse a room description into structured data",
                "input_schema": ParsedRoomDescription.model_json_schema(),
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=ROOM_PARSING_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "tool", "name": "parse_room"},
            messages=[{"role": "user", "content": description}],
        )

        # Extract the tool use content
        for block in response.content:
            if block.type == "tool_use" and block.name == "parse_room":
                return ParsedRoomDescription.model_validate(block.input)

        # Fallback: should not reach here with tool_choice
        raise ValueError("Failed to extract structured room data from Claude response")

    def _normalize_category(self, category: str) -> str:
        """Map colloquial terms to catalog categories.

        Args:
            category: Furniture category from LLM extraction

        Returns:
            Normalized category name for catalog lookup
        """
        return CATEGORY_ALIASES.get(category.lower(), category.lower())

    def _select_furniture(
        self, requests: List[RawFurnitureRequest], seed: int
    ) -> List[FurnitureSpec]:
        """Select models from catalog for each furniture request.

        Args:
            requests: List of furniture requests from LLM extraction
            seed: Random seed for reproducibility

        Returns:
            List of FurnitureSpec objects with catalog models
        """
        rng = np.random.default_rng(seed)
        specs: List[FurnitureSpec] = []

        for req in requests:
            category = self._normalize_category(req.category)

            for _ in range(req.quantity):
                try:
                    model = self.catalog.get_random_model(category, req.style, rng)
                    dims = self.catalog.get_dimensions(model["model_id"])
                    specs.append(
                        FurnitureSpec(
                            category=category,
                            model_id=model["model_id"],
                            model_path=str(
                                self.catalog.get_model_path(model["model_id"])
                            ),
                            dimensions=dims,
                            preferred_wall=req.preferred_wall,
                        )
                    )
                except ValueError:
                    # No models found for this category - skip silently
                    # This can happen if the catalog doesn't have the category
                    pass

        return specs

    def _place_windows(self, room: Room, room_type: Optional[str]) -> Room:
        """Place windows based on room type and building codes.

        Args:
            room: Room with furniture layout (from optimize_layout)
            room_type: Type of room ("bedroom", "living_room", etc.)

        Returns:
            New Room instance with windows added (or original if no windows needed)
        """
        if not room_type:
            return room

        # Normalize: lowercase, replace spaces with underscores
        room_type_normalized = room_type.lower().replace(" ", "_")
        placer = WindowPlacer()

        if room_type_normalized == "bedroom":
            return placer.place_for_bedroom(room)
        elif room_type_normalized in ("living_room", "living", "family_room"):
            return placer.place_for_living(room)

        # office, other, or unknown -> no automatic windows
        return room
