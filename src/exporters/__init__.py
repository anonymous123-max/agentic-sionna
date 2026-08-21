"""Room layout exporters.

Provides exporters for converting Room models to various output formats:
- PNGExporter: Floor plan PNG images
- ASCIIExporter: Terminal-friendly ASCII art
- XMLExporter: Sionna-compatible Mitsuba 3.0 XML for ray tracing
- GLTFExporter: GLTF/GLB 3D scene export

Also provides alignment validation utilities:
- validate_alignment: Cross-format position consistency checker
- extract_positions_xml: Extract positions from Sionna XML
- extract_positions_gltf: Extract positions from GLTF/GLB
- extract_positions_ascii: Extract positions from ASCII art
"""

from src.exporters.ascii import ASCIIExporter
from src.exporters.base import Exporter
from src.exporters.gltf import GLTFExporter
from src.exporters.png import PNGExporter
from src.exporters.validator import (
    extract_positions_ascii,
    extract_positions_gltf,
    extract_positions_xml,
    validate_alignment,
)
from src.exporters.materials import ITU_MATERIALS
from src.exporters.xml import XMLExporter

__all__ = [
    "ASCIIExporter",
    "Exporter",
    "GLTFExporter",
    "ITU_MATERIALS",
    "PNGExporter",
    "XMLExporter",
    "extract_positions_ascii",
    "extract_positions_gltf",
    "extract_positions_xml",
    "validate_alignment",
]
