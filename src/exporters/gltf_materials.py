"""GLTF material colors and PBR material management.

Provides material-to-color mapping, PBR material caching, and material
consolidation for GLTF/GLB 3D scene exports.
"""

import trimesh
from trimesh.visual.material import PBRMaterial


# Material to RGBA color mapping for Blender visualization
# Colors chosen to visually represent ITU radio materials
MATERIAL_COLORS = {
    # Ground materials
    'grass': [76, 153, 0, 255],       # Green
    'wet_ground': [101, 67, 33, 255], # Brown
    'concrete': [180, 180, 180, 255], # Gray
    'asphalt': [60, 60, 60, 255],     # Dark gray

    # Building materials
    'brick': [178, 102, 68, 255],     # Red-brown
    'glass': [173, 216, 230, 180],    # Light blue, semi-transparent
    'metal': [192, 192, 192, 255],    # Silver
    'wood': [139, 90, 43, 255],       # Brown
    'plasterboard': [245, 240, 230, 255],  # Off-white
    'ceiling_board': [250, 245, 240, 255], # Light cream

    # Special elements
    'tree_trunk': [101, 67, 33, 255], # Brown
    'tree_crown': [34, 139, 34, 255], # Forest green
    'road': [50, 50, 50, 255],        # Asphalt dark gray
    'wall': [235, 230, 220, 255],     # Light beige
    'floor': [210, 180, 140, 255],    # Tan

    # Default fallback
    '_default': [150, 150, 150, 255], # Medium gray
}


def get_color_for_material(material: str) -> list:
    """Get RGBA color for a material name.

    Args:
        material: ITU material name or mesh type

    Returns:
        RGBA color as list of 4 integers [0-255]
    """
    mat_lower = material.lower().strip()
    if mat_lower in MATERIAL_COLORS:
        return MATERIAL_COLORS[mat_lower]
    return MATERIAL_COLORS['_default']


# Cache for PBR materials to avoid duplicates in GLTF export
_material_cache: dict[str, PBRMaterial] = {}


def get_pbr_material(material: str) -> PBRMaterial:
    """Get or create a cached PBR material for the given ITU material name.

    Caching ensures the same material instance is reused across meshes,
    preventing GLTF from creating duplicates like 'metal.001'.

    Args:
        material: ITU material name (e.g., 'concrete', 'glass', 'wood', 'metal')

    Returns:
        Cached PBRMaterial instance
    """
    if material not in _material_cache:
        rgba = get_color_for_material(material)
        # Convert 0-255 to 0-1 range for PBR
        base_color = [c / 255.0 for c in rgba]

        # Create PBR material with exact ITU material name for Sionna compatibility
        _material_cache[material] = PBRMaterial(
            name=material,  # Use exact ITU material name
            baseColorFactor=base_color,
            metallicFactor=0.1 if material == 'metal' else 0.0,
            roughnessFactor=0.3 if material == 'glass' else 0.8,
        )

    return _material_cache[material]


def clear_material_cache() -> None:
    """Clear the material cache. Call before each export for clean material names."""
    _material_cache.clear()


def apply_material_color(mesh: trimesh.Trimesh, material: str) -> None:
    """Apply PBR material with color to mesh for proper GLTF export.

    Uses cached materials to ensure exact ITU material names without suffixes.

    Args:
        mesh: Trimesh to apply material to
        material: ITU material name (e.g., 'concrete', 'glass', 'wood', 'metal')
    """
    pbr = get_pbr_material(material)
    mesh.visual = trimesh.visual.TextureVisuals(material=pbr)


# Simplified furniture category to material mapping for GLTF colors
# Maps 3D-FUTURE categories to ITU material types
FURNITURE_CATEGORY_MATERIALS = {
    # Seating - mostly wood/fabric (use wood for structural)
    'chair': 'wood', 'armchair': 'wood', 'sofa': 'wood', 'stool': 'wood',
    'lounge chair': 'wood', 'dining chair': 'wood', 'chaise longue': 'wood',
    # Tables - wood
    'table': 'wood', 'desk': 'wood', 'coffee table': 'wood', 'dining table': 'wood',
    'console table': 'wood', 'side table': 'wood', 'corner/side table': 'wood',
    # Storage - wood
    'cabinet': 'wood', 'shelf': 'wood', 'bookcase': 'wood', 'wardrobe': 'wood',
    'dresser': 'wood', 'nightstand': 'wood', 'tv stand': 'wood', 'wine cabinet': 'wood',
    'shoe cabinet': 'wood', 'sideboard': 'wood', 'drawer chest': 'wood',
    # Beds - wood frame
    'bed': 'wood', 'kids bed': 'wood', 'bunk bed': 'wood',
    # Lighting - metal
    'lamp': 'metal', 'ceiling lamp': 'metal', 'pendant lamp': 'metal',
    'floor lamp': 'metal', 'wall lamp': 'metal',
    # Metal items
    'appliance': 'metal', 'kitchen appliance': 'metal',
    # Glass items
    'mirror': 'glass',
}


def get_color_for_category(category: str) -> list:
    """Get RGBA color for a furniture category.

    Args:
        category: 3D-FUTURE furniture category

    Returns:
        RGBA color as list of 4 integers [0-255]
    """
    cat_lower = category.lower().strip()

    # Direct match
    if cat_lower in FURNITURE_CATEGORY_MATERIALS:
        return get_color_for_material(FURNITURE_CATEGORY_MATERIALS[cat_lower])

    # Partial match
    for key, material in FURNITURE_CATEGORY_MATERIALS.items():
        if key in cat_lower or cat_lower in key:
            return get_color_for_material(material)

    # Keyword fallback
    if any(kw in cat_lower for kw in ['bed', 'sofa', 'chair', 'table', 'desk', 'cabinet', 'shelf']):
        return get_color_for_material('wood')
    if any(kw in cat_lower for kw in ['lamp', 'light']):
        return get_color_for_material('metal')
    if any(kw in cat_lower for kw in ['mirror', 'glass']):
        return get_color_for_material('glass')

    # Default wood (most furniture is wood)
    return get_color_for_material('wood')


def _consolidate_materials(tm_scene: trimesh.Scene) -> None:
    """Consolidate materials in scene to avoid duplicates like 'wood.001'.

    GLTF export creates duplicate materials if meshes have different material
    objects even with the same name. This function ensures all meshes with
    the same material name share the exact same material object.

    Args:
        tm_scene: trimesh Scene to consolidate materials in
    """
    # Collect unique materials by name
    material_by_name: dict[str, PBRMaterial] = {}

    for mesh in tm_scene.geometry.values():
        if not hasattr(mesh, 'visual') or mesh.visual is None:
            continue
        if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
            continue

        mat = mesh.visual.material
        mat_name = getattr(mat, 'name', None)
        if mat_name and mat_name not in material_by_name:
            material_by_name[mat_name] = mat

    # Apply shared materials to all meshes
    for mesh in tm_scene.geometry.values():
        if not hasattr(mesh, 'visual') or mesh.visual is None:
            continue
        if not hasattr(mesh.visual, 'material') or mesh.visual.material is None:
            continue

        mat = mesh.visual.material
        mat_name = getattr(mat, 'name', None)
        if mat_name and mat_name in material_by_name:
            # Replace with shared material instance
            mesh.visual.material = material_by_name[mat_name]
