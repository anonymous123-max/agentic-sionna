"""Session management for multi-turn conversation.

Provides state tracking and error handling for iterative layout refinement.
"""

from src.session.errors import ErrorCode, RoomLayoutError
from src.session.refinement import (
    RefinementCommand,
    execute_command,
    optimize_with_locked,
)
from src.session.state import ConversationState

__all__ = [
    "ConversationState",
    "ErrorCode",
    "RefinementCommand",
    "RoomLayoutError",
    "execute_command",
    "optimize_with_locked",
]
