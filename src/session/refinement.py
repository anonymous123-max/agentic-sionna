"""Refinement workflow for multi-turn conversation.

Provides command parsing and execution for iterative layout optimization,
allowing users to lock, unlock, regenerate, add, and remove furniture.
"""

from typing import List, Literal

from pydantic import BaseModel

from src.models.room import FurnitureItem, Room
from src.optimizer.layout import FurnitureSpec, LayoutOptimizer
from src.session.errors import ErrorCode, RoomLayoutError
from src.session.state import ConversationState


class RefinementCommand(BaseModel):
    """User command for layout refinement.

    Attributes:
        action: Type of refinement action to perform
        furniture_ids: List of furniture IDs (for lock/unlock/remove)
        description: Natural language description (for add action)
    """

    action: Literal["lock", "unlock", "regenerate", "add", "remove"]
    furniture_ids: List[str] = []
    description: str | None = None


def optimize_with_locked(
    state: ConversationState,
    catalog: "FutureCatalog | None" = None,  # noqa: F821
    new_specs: List[FurnitureSpec] | None = None,
) -> Room:
    """Re-optimize layout with locked furniture as obstacles.

    Locked furniture remains in place while unlocked furniture is
    re-optimized around them. Optionally adds new furniture specs.

    Args:
        state: Current conversation state with room and locked IDs
        catalog: Optional FutureCatalog for model paths (not used currently)
        new_specs: Optional new furniture specs to add

    Returns:
        New Room with locked + re-optimized furniture
    """
    locked_items = state.get_locked_furniture()
    unlocked_items = state.get_unlocked_furniture()

    # Convert unlocked items to FurnitureSpec for re-optimization
    unlocked_specs = [
        FurnitureSpec(
            category=item.category,
            model_id=item.model_id,
            model_path=item.model_path,
            dimensions=item.dimensions,
        )
        for item in unlocked_items
    ]

    # Add any new specs
    if new_specs:
        unlocked_specs.extend(new_specs)

    # Create optimizer with room dimensions
    optimizer = LayoutOptimizer(
        room_width=state.room.width,
        room_length=state.room.length,
        doors=list(state.room.doors) if state.room.doors else [],
    )

    # Optimize unlocked furniture with locked as obstacles
    new_furniture = optimizer.optimize_with_obstacles(
        furniture_specs=unlocked_specs,
        fixed_obstacles=locked_items,
    )

    # Combine locked and newly optimized furniture
    all_furniture = locked_items + new_furniture

    return Room(
        width=state.room.width,
        length=state.room.length,
        height=state.room.height,
        doors=list(state.room.doors) if state.room.doors else [],
        windows=list(state.room.windows) if state.room.windows else [],
        furniture=all_furniture,
    )


def execute_command(
    state: ConversationState,
    command: RefinementCommand,
    catalog: "FutureCatalog | None" = None,  # noqa: F821
) -> ConversationState:
    """Execute a refinement command on the conversation state.

    Handles lock, unlock, regenerate, remove, and add actions.
    All exceptions are wrapped in RoomLayoutError for user-friendly display.

    Args:
        state: Current conversation state
        command: Refinement command to execute
        catalog: Optional FutureCatalog (required for "add" action)

    Returns:
        New ConversationState after executing command

    Raises:
        RoomLayoutError: On any error, with actionable message
    """
    try:
        if command.action == "lock":
            return state.lock_furniture(command.furniture_ids)

        if command.action == "unlock":
            return state.unlock_furniture(command.furniture_ids)

        if command.action == "regenerate":
            new_room = optimize_with_locked(state, catalog=catalog)
            return state.update_room(new_room)

        if command.action == "remove":
            # Filter out the specified furniture IDs
            remaining = [
                item
                for item in state.room.furniture
                if item.id not in command.furniture_ids
            ]
            new_room = Room(
                width=state.room.width,
                length=state.room.length,
                height=state.room.height,
                doors=list(state.room.doors) if state.room.doors else [],
                windows=list(state.room.windows) if state.room.windows else [],
                furniture=remaining,
            )
            # Also remove from locked_ids
            new_locked = state.locked_ids - frozenset(command.furniture_ids)
            new_state = ConversationState(
                room=new_room,
                locked_ids=new_locked,
                turn_count=state.turn_count + 1,
            )
            return new_state

        if command.action == "add":
            raise RoomLayoutError(
                code=ErrorCode.PARSE_FAILED,
                message="Add action requires catalog and description",
                suggestion="Provide a FutureCatalog and description for furniture to add",
            )

        # Should not reach here due to Literal type, but handle anyway
        raise RoomLayoutError(
            code=ErrorCode.PARSE_FAILED,
            message=f"Unknown action: {command.action}",
            suggestion="Valid actions are: lock, unlock, regenerate, add, remove",
        )

    except RoomLayoutError:
        # Re-raise our own errors
        raise
    except Exception as e:
        # Wrap unexpected errors
        raise RoomLayoutError(
            code=ErrorCode.OPTIMIZATION_FAILED,
            message=f"Refinement failed: {str(e)}",
            suggestion="Check the furniture IDs and try again",
            details={"original_error": type(e).__name__},
        ) from e
