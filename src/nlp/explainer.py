"""Placement explainer for generating human-readable furniture rationales.

Uses Claude API to generate natural language explanations for furniture
placements based on constraint costs from the optimizer.
"""

from pydantic import BaseModel, Field
from anthropic import Anthropic

from src.models.room import Room
from src.optimizer.layout import LayoutOptimizer, PlacementDecision


class FurnitureExplanation(BaseModel):
    """Explanation for a single furniture placement.

    Attributes:
        item_id: Furniture item ID
        category: Furniture category
        explanation: Human-readable placement rationale
    """

    item_id: str = Field(description="Furniture item ID")
    category: str = Field(description="Furniture category (bed, desk, etc.)")
    explanation: str = Field(description="Human-readable placement rationale")


class PlacementExplanations(BaseModel):
    """All placement explanations for a room.

    Attributes:
        placements: List of explanations for each furniture piece
        summary: Overall layout summary in 1-2 sentences
    """

    placements: list[FurnitureExplanation] = Field(
        description="Explanations for each furniture piece"
    )
    summary: str = Field(description="Overall layout summary in 1-2 sentences")


EXPLANATION_SYSTEM_PROMPT = """You are explaining furniture placement decisions to a user.

Each placement was determined by an optimization algorithm balancing these constraints:
- wall_affinity (weight=2): Furniture backs should be against walls for a natural look
- collision (weight=10): No overlapping furniture (highest priority safety constraint)
- pathway (weight=3): Doors must have clearance for walking
- in_room (weight=5): Furniture must be fully inside room bounds

Given the constraint costs for each item, explain WHY it was placed where it is.
Use natural language that a non-technical user would understand.
Focus on practical reasons: "The bed is against the north wall to leave space for the door."
Do NOT mention specific coordinates or constraint weights.

Call the explain_placements tool with your explanations."""


class PlacementExplainer:
    """Generate human-readable explanations for furniture placements.

    Uses Claude API to convert constraint costs into natural language
    rationales that explain why each piece of furniture was placed
    where it is.

    Example:
        explainer = PlacementExplainer()
        explanations = explainer.explain_room(room)
        for exp in explanations.placements:
            print(f"{exp.category}: {exp.explanation}")
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-5-20250514",
    ):
        """Initialize explainer with API client.

        Args:
            api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if None)
            model: Claude model to use for explanation generation
        """
        self.client = Anthropic(api_key=api_key)
        self.model = model

    def explain(
        self,
        room: Room,
        decisions: list[PlacementDecision],
    ) -> PlacementExplanations:
        """Generate explanations for all furniture placements.

        Args:
            room: The optimized Room
            decisions: PlacementDecision list from optimizer

        Returns:
            PlacementExplanations with rationale for each piece
        """
        # Build context for LLM
        placement_data = [
            {
                "item": d.category,
                "id": d.furniture_id,
                "position": f"({d.position[0]:.1f}m, {d.position[1]:.1f}m)",
                "dominant_constraint": d.dominant_constraint,
                "costs": {k: round(v, 2) for k, v in d.constraint_costs.items()},
            }
            for d in decisions
        ]

        # Build door descriptions
        door_descriptions = [f"{d.wall} wall" for d in room.doors]

        context = f"""Room: {room.width}m x {room.length}m
Doors: {door_descriptions}

Furniture placements:
{placement_data}

Explain each placement in natural language."""

        # Define tool for structured extraction (same pattern as other NLP modules)
        tools = [
            {
                "name": "explain_placements",
                "description": "Provide explanations for furniture placements",
                "input_schema": PlacementExplanations.model_json_schema(),
            }
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=EXPLANATION_SYSTEM_PROMPT,
            tools=tools,
            tool_choice={"type": "tool", "name": "explain_placements"},
            messages=[{"role": "user", "content": context}],
        )

        # Extract the tool use content
        for block in response.content:
            if block.type == "tool_use" and block.name == "explain_placements":
                return PlacementExplanations.model_validate(block.input)

        # Fallback: should not reach here with tool_choice
        raise ValueError("Failed to generate placement explanations")

    def explain_room(self, room: Room) -> PlacementExplanations:
        """Convenience method: create optimizer, get decisions, explain.

        Creates a LayoutOptimizer to calculate constraint costs for each
        furniture item, then generates explanations.

        Args:
            room: Optimized Room with furniture

        Returns:
            PlacementExplanations
        """
        optimizer = LayoutOptimizer(room.width, room.length, room.doors)
        decisions = optimizer.get_placement_decisions(room.furniture)
        return self.explain(room, decisions)
