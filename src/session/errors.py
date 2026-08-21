"""Error handling with user-friendly formatting.

Provides custom exceptions for room layout operations with actionable
error messages and suggestions for resolution.
"""

from enum import Enum
from typing import Any


class ErrorCode(Enum):
    """Error codes for room layout operations.

    Used to categorize errors for consistent handling and messaging.
    """

    INVALID_DIMENSIONS = "invalid_dimensions"
    FURNITURE_NOT_FOUND = "furniture_not_found"
    OPTIMIZATION_FAILED = "optimization_failed"
    API_ERROR = "api_error"
    CATALOG_ERROR = "catalog_error"
    ROOM_TOO_SMALL = "room_too_small"
    PARSE_FAILED = "parse_failed"


# Mapping from Pydantic error types to user-friendly suggestions
_ERROR_TYPE_SUGGESTIONS = {
    "greater_than": "Value must be positive",
    "greater_than_equal": "Value must be positive or zero",
    "less_than": "Value exceeds maximum allowed",
    "missing": "This is a required field",
    "string_type": "Expected text value",
    "int_type": "Expected whole number",
    "float_type": "Expected numeric value",
    "bool_type": "Expected true/false value",
    "value_error": "Invalid value provided",
}


class RoomLayoutError(Exception):
    """Custom exception for room layout operations.

    Provides user-friendly error messages with optional suggestions
    and details for debugging.

    Attributes:
        code: ErrorCode categorizing this error
        message: Human-readable error description
        suggestion: Optional action the user can take
        details: Optional dict with additional context
    """

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        suggestion: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize RoomLayoutError.

        Args:
            code: ErrorCode categorizing this error
            message: Human-readable error description
            suggestion: Optional action the user can take
            details: Optional dict with additional context
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestion = suggestion
        self.details = details or {}

    def to_user_string(self) -> str:
        """Format error for user display.

        Returns:
            Formatted string with error message and suggestion
        """
        result = f"Error: {self.message}"
        if self.suggestion:
            result += f"\nSuggestion: {self.suggestion}"
        return result

    @classmethod
    def from_validation_error(cls, error: Exception) -> "RoomLayoutError":
        """Create RoomLayoutError from Pydantic ValidationError.

        Extracts the first error from validation results and transforms
        it into a user-friendly error message with suggestions.

        Args:
            error: Pydantic ValidationError (v2 format)

        Returns:
            RoomLayoutError with appropriate code and suggestion
        """
        # Pydantic v2: error.errors() returns list of dicts
        errors = error.errors()  # type: ignore[union-attr]

        if not errors:
            return cls(
                code=ErrorCode.INVALID_DIMENSIONS,
                message="Validation failed",
                suggestion="Check input values",
            )

        first_error = errors[0]

        # Build field path from loc tuple
        loc = first_error.get("loc", ())
        field_path = ".".join(str(part) for part in loc) if loc else "input"

        # Get error type and message
        error_type = first_error.get("type", "value_error")
        error_msg = first_error.get("msg", "Invalid value")

        # Look up suggestion based on error type
        suggestion = _ERROR_TYPE_SUGGESTIONS.get(
            error_type,
            f"Check the value for '{field_path}'",
        )

        return cls(
            code=ErrorCode.INVALID_DIMENSIONS,
            message=f"Invalid value for '{field_path}': {error_msg}",
            suggestion=suggestion,
            details={"field": field_path, "error_type": error_type},
        )

    @classmethod
    def from_api_error(cls, error: Exception) -> "RoomLayoutError":
        """Create RoomLayoutError from Anthropic SDK exceptions.

        Handles APIError, RateLimitError, and APIConnectionError from
        the anthropic SDK with appropriate user-friendly messages.

        Args:
            error: Exception from anthropic SDK

        Returns:
            RoomLayoutError with appropriate code and suggestion
        """
        error_name = type(error).__name__

        if error_name == "RateLimitError":
            return cls(
                code=ErrorCode.API_ERROR,
                message="API rate limit exceeded",
                suggestion="Wait a moment and try again",
                details={"exception_type": error_name},
            )

        if error_name == "APIConnectionError":
            return cls(
                code=ErrorCode.API_ERROR,
                message="Unable to connect to API",
                suggestion="Check your internet connection and try again",
                details={"exception_type": error_name},
            )

        if error_name == "AuthenticationError":
            return cls(
                code=ErrorCode.API_ERROR,
                message="API authentication failed",
                suggestion="Check your API key configuration",
                details={"exception_type": error_name},
            )

        # Generic APIError or other anthropic exceptions
        return cls(
            code=ErrorCode.API_ERROR,
            message=f"API error: {str(error)}",
            suggestion="Try again or check API status",
            details={"exception_type": error_name},
        )
