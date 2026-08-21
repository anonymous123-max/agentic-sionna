"""Configuration package for sionna-skill.

Provides SkillConfig singleton for centralized configuration management,
including 3D-FUTURE dataset path discovery.
"""

from .skill_config import SkillConfig

__all__ = ["SkillConfig"]
