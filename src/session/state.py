"""Conversation state for multi-turn refinement.

Provides immutable state tracking for iterative layout optimization,
allowing users to lock furniture positions across refinement turns.
"""

from dataclasses import dataclass, field
from typing import FrozenSet, List

from src.models.room import FurnitureItem, Room


@dataclass(frozen=True)
class ConversationState:
    """Immutable state for multi-turn conversation.

    Tracks which furniture items are locked (fixed) during re-optimization,
    enabling iterative refinement where users can approve some placements
    while requesting changes to others.

    The frozen dataclass ensures state immutability - all mutation methods
    return new state instances rather than modifying in-place.

    Attributes:
        room: Current room layout with furniture
        locked_ids: Set of furniture IDs that should not be moved
        turn_count: Number of refinement turns completed
    """

    room: Room
    locked_ids: FrozenSet[str] = field(default_factory=frozenset)
    turn_count: int = 0

    def lock_furniture(self, furniture_ids: List[str]) -> "ConversationState":
        """Lock furniture items to prevent re-optimization.

        Args:
            furniture_ids: List of furniture IDs to lock

        Returns:
            New ConversationState with IDs added to locked_ids
        """
        new_locked = self.locked_ids | frozenset(furniture_ids)
        return ConversationState(
            room=self.room,
            locked_ids=new_locked,
            turn_count=self.turn_count + 1,
        )

    def unlock_furniture(self, furniture_ids: List[str]) -> "ConversationState":
        """Unlock furniture items to allow re-optimization.

        Args:
            furniture_ids: List of furniture IDs to unlock

        Returns:
            New ConversationState with IDs removed from locked_ids
        """
        new_locked = self.locked_ids - frozenset(furniture_ids)
        return ConversationState(
            room=self.room,
            locked_ids=new_locked,
            turn_count=self.turn_count + 1,
        )

    def update_room(self, new_room: Room) -> "ConversationState":
        """Update room layout while preserving locked IDs.

        Args:
            new_room: New room layout after re-optimization

        Returns:
            New ConversationState with updated room
        """
        return ConversationState(
            room=new_room,
            locked_ids=self.locked_ids,
            turn_count=self.turn_count + 1,
        )

    def get_locked_furniture(self) -> List[FurnitureItem]:
        """Get list of locked furniture items.

        Returns:
            List of FurnitureItem objects whose id is in locked_ids
        """
        return [item for item in self.room.furniture if item.id in self.locked_ids]

    def get_unlocked_furniture(self) -> List[FurnitureItem]:
        """Get list of unlocked furniture items.

        Returns:
            List of FurnitureItem objects whose id is NOT in locked_ids
        """
        return [item for item in self.room.furniture if item.id not in self.locked_ids]
