"""Skill configuration with 3D-FUTURE dataset discovery.

Provides singleton configuration for the sionna-skill package with support for
discovering the 3D-FUTURE dataset path from multiple sources.

Discovery priority:
1. FUTURE_DATASET_PATH environment variable
2. ~/.sionna-skill/config.json with "future_dataset_path" key
3. Common locations (~/Downloads/3D-FUTURE-model, ~/3D-FUTURE-model, /data/3D-FUTURE-model)

Usage:
    from src.config import SkillConfig
    config = SkillConfig()
    path = config.future_dataset_path  # Raises FileNotFoundError if not found
"""

import json
import os
from pathlib import Path
from typing import Optional


class SkillConfig:
    """Singleton configuration for sionna-skill.

    Provides lazy discovery of 3D-FUTURE dataset path with validation.
    Uses singleton pattern to ensure consistent configuration across the application.

    Attributes:
        future_dataset_path: Path to 3D-FUTURE dataset (lazily discovered)

    Example:
        >>> config = SkillConfig()
        >>> path = config.future_dataset_path
        >>> print(path)
        PosixPath('/Users/user/Downloads/3D-FUTURE-model')
    """

    _instance: Optional["SkillConfig"] = None

    def __new__(cls) -> "SkillConfig":
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize configuration (only runs once due to singleton)."""
        if self._initialized:
            return
        self._initialized = True
        self._future_path: Optional[Path] = None

    @property
    def future_dataset_path(self) -> Path:
        """Get path to 3D-FUTURE dataset.

        Lazily discovers and validates the path on first access.
        Caches the result for subsequent accesses.

        Returns:
            Path to validated 3D-FUTURE dataset directory

        Raises:
            FileNotFoundError: If dataset cannot be found in any location
        """
        if self._future_path is None:
            self._future_path = self._discover_future_path()
        return self._future_path

    def _discover_future_path(self) -> Path:
        """Discover 3D-FUTURE dataset path from multiple sources.

        Priority:
        1. FUTURE_DATASET_PATH environment variable
        2. ~/.sionna-skill/config.json with "future_dataset_path" key
        3. Common locations

        Returns:
            Path to validated 3D-FUTURE dataset

        Raises:
            FileNotFoundError: If no valid path found
        """
        # Priority 1: Environment variable
        env_path = os.environ.get("FUTURE_DATASET_PATH")
        if env_path:
            path = Path(env_path).expanduser()
            if self._validate_future_path(path):
                return path

        # Priority 2: Config file
        config_file = Path.home() / ".sionna-skill" / "config.json"
        if config_file.exists():
            try:
                with open(config_file) as f:
                    config = json.load(f)
                if "future_dataset_path" in config:
                    path = Path(config["future_dataset_path"]).expanduser()
                    if self._validate_future_path(path):
                        return path
            except (json.JSONDecodeError, OSError):
                # Config file exists but is invalid - continue to next source
                pass

        # Priority 3: Common locations
        common_locations = [
            Path.home() / "Downloads" / "3D-FUTURE-model",
            Path.home() / "3D-FUTURE-model",
            Path("/data/3D-FUTURE-model"),
        ]
        for path in common_locations:
            if self._validate_future_path(path):
                return path

        # Priority 4: Not found - raise helpful error
        raise FileNotFoundError(
            "3D-FUTURE dataset not found. Please either:\n"
            "1. Set FUTURE_DATASET_PATH environment variable\n"
            "2. Create ~/.sionna-skill/config.json with "
            '{"future_dataset_path": "/path/to/3D-FUTURE-model"}\n'
            "\n"
            "The dataset should contain a 'model_info.json' file."
        )

    def _validate_future_path(self, path: Path) -> bool:
        """Validate that path is a valid 3D-FUTURE dataset directory.

        Args:
            path: Path to validate

        Returns:
            True if path exists, is a directory, and contains model_info.json
        """
        if not path.exists():
            return False
        if not path.is_dir():
            return False
        # 3D-FUTURE dataset should contain model_info.json
        model_info = path / "model_info.json"
        return model_info.exists()
