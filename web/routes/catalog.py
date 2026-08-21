"""Furniture catalog routes: categories, search, mesh, GLB, image."""

import logging
from pathlib import Path

import numpy as np
from flask import Blueprint, Response, jsonify, request, send_file

from routes.shared import GENERIC_FURNITURE_DIMS, _get_catalog

logger = logging.getLogger(__name__)

catalog_bp = Blueprint("catalog", __name__)


@catalog_bp.route("/api/catalog/categories")
def catalog_categories():
    """Return available 3D-FUTURE furniture categories."""
    catalog = _get_catalog()
    if catalog is None:
        # Return generic categories when catalog unavailable
        return jsonify({
            "available": False,
            "categories": [
                {"name": cat, "count": 0}
                for cat in GENERIC_FURNITURE_DIMS.keys()
            ],
        })

    categories = []
    for name, model_ids in sorted(catalog.category_index.items()):
        categories.append({"name": name, "count": len(model_ids)})

    return jsonify({"available": True, "categories": categories})


@catalog_bp.route("/api/catalog/search")
def catalog_search():
    """Search furniture models by query string (category/style/material)."""
    q = (request.args.get("q", "") or "").strip().lower()
    limit = min(int(request.args.get("limit", 50)), 200)
    catalog = _get_catalog()

    if not q:
        return catalog_categories()

    if catalog is None:
        # Filter generic categories by query
        matches = [
            {"name": cat, "count": 0}
            for cat in GENERIC_FURNITURE_DIMS
            if q in cat
        ]
        return jsonify({"available": False, "categories": matches})

    # Search across categories, styles, materials
    results = []
    seen_cats = set()
    for name, model_ids in sorted(catalog.category_index.items()):
        if q in name:
            results.append({"name": name, "count": len(model_ids)})
            seen_cats.add(name)

    # Also search individual model metadata
    for model_id, model in catalog.models.items():
        cat = (model.get("category") or "").lower()
        style = (model.get("style") or "").lower()
        material = (model.get("material") or "").lower()
        theme = (model.get("theme") or "").lower()
        if cat in seen_cats:
            continue
        if q in style or q in material or q in theme:
            if cat and cat not in seen_cats:
                count = len(catalog.category_index.get(cat, []))
                results.append({"name": cat, "count": count})
                seen_cats.add(cat)
        if len(results) >= limit:
            break

    return jsonify({"available": True, "categories": results[:limit]})


@catalog_bp.route("/api/catalog/model/<model_id>/mesh")
def catalog_model_mesh(model_id: str):
    """Return OBJ mesh vertices and faces for a 3D-FUTURE model.

    Applies Y-up to Z-up coordinate transform.
    Decimates to max_faces if model is very detailed.
    """
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"has_mesh": False, "error": "Catalog unavailable"}), 404

    try:
        import trimesh
        if hasattr(catalog, 'get_model_file'):
            mesh_file = catalog.get_model_file(model_id)
        else:
            mesh_file = catalog.get_model_path(model_id) / "raw_model.obj"
        if not Path(str(mesh_file)).exists():
            return jsonify({"has_mesh": False, "error": "Model file not found"}), 404

        mesh = trimesh.load(str(mesh_file), force="mesh")

        # Decimate if too many faces for browser rendering
        max_faces = int(request.args.get("max_faces", 500000))
        if len(mesh.faces) > max_faces:
            try:
                mesh = mesh.simplify_quadric_decimation(max_faces)
            except (ImportError, AttributeError):
                # fast_simplification not installed; subsample faces instead
                step = max(1, len(mesh.faces) // max_faces)
                keep = mesh.faces[::step]
                used_verts = np.unique(keep.flatten())
                remap = np.full(len(mesh.vertices), -1, dtype=int)
                remap[used_verts] = np.arange(len(used_verts))
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices[used_verts],
                    faces=remap[keep],
                )

        verts = mesh.vertices.copy()

        # Y-up to Z-up: rotate -90 deg around X
        # new_y = -old_z, new_z = old_y
        verts_transformed = np.column_stack([
            verts[:, 0],      # X stays
            -verts[:, 2],     # new Y = -old Z
            verts[:, 1],      # new Z = old Y (height)
        ])

        # Center on origin (XY), keep Z base at 0
        mins = verts_transformed.min(axis=0)
        maxs = verts_transformed.max(axis=0)
        center_x = (mins[0] + maxs[0]) / 2
        center_y = (mins[1] + maxs[1]) / 2
        verts_transformed[:, 0] -= center_x
        verts_transformed[:, 1] -= center_y
        verts_transformed[:, 2] -= mins[2]  # base at z=0

        faces = mesh.faces.tolist()

        return jsonify({
            "has_mesh": True,
            "model_id": model_id,
            "vertices": verts_transformed.tolist(),
            "faces": faces,
            "num_vertices": len(verts_transformed),
            "num_faces": len(faces),
        })
    except Exception as e:
        return jsonify({"has_mesh": False, "error": str(e)}), 500


@catalog_bp.route("/api/catalog/model/<model_id>/image")
def catalog_model_image(model_id: str):
    """Return the preview image (image.jpg) for a 3D-FUTURE model."""
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"error": "Catalog unavailable"}), 404

    try:
        model_dir = catalog.get_model_path(model_id)
        image_path = model_dir / "image.jpg"
        if not image_path.exists():
            return jsonify({"error": "Image not found"}), 404
        resp = send_file(str(image_path), mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "public, max-age=86400"
        return resp
    except (KeyError, ValueError) as e:
        logger.debug("Model image not found for %s: %s", model_id, e)
        return jsonify({"error": "Model not found"}), 404


@catalog_bp.route("/api/catalog/category/<name>/models")
def catalog_category_models(name: str):
    """Return individual models within a furniture category."""
    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"error": "Catalog unavailable"}), 404

    limit = min(int(request.args.get("limit", 30)), 100)
    q = (request.args.get("q", "") or "").strip().lower()

    model_ids = catalog.category_index.get(name, [])
    results = []
    for mid in model_ids:
        model = catalog.models.get(mid, {})
        style = (model.get("style") or "")
        material = (model.get("material") or "")
        theme = (model.get("theme") or "")

        if q:
            searchable = f"{style} {material} {theme} {name}".lower()
            if q not in searchable:
                continue

        results.append({
            "model_id": mid,
            "category": name,
            "style": style,
            "material": material,
            "theme": theme,
            "super_category": model.get("super-category", ""),
        })
        if len(results) >= limit:
            break

    return jsonify({"category": name, "models": results, "total": len(model_ids)})


GLB_CACHE_DIR = Path("data/glb-cache")


@catalog_bp.route("/api/catalog/model/<model_id>/glb")
def catalog_model_glb(model_id: str):
    """Return a binary GLB file for a 3D-FUTURE model with textures.

    Uses a disk cache to avoid repeated trimesh processing.
    Preserves materials and textures by exporting the Scene directly
    (not concatenating into a single mesh). Only decimates individual
    geometries that exceed the face budget.
    """
    # Disk cache: serve pre-built GLB if available
    cache_path = GLB_CACHE_DIR / f"{model_id}.glb"
    if cache_path.exists():
        resp = send_file(str(cache_path), mimetype="model/gltf-binary")
        resp.headers["Cache-Control"] = "public, max-age=604800, immutable"
        return resp

    catalog = _get_catalog()
    if catalog is None:
        return jsonify({"error": "Catalog unavailable"}), 404

    try:
        import trimesh

        if hasattr(catalog, 'get_model_file'):
            mesh_file = Path(str(catalog.get_model_file(model_id)))
        else:
            mesh_file = catalog.get_model_path(model_id) / "raw_model.obj"
        if not mesh_file.exists():
            return jsonify({"error": "Model file not found"}), 404

        scene_or_mesh = trimesh.load(str(mesh_file))

        max_total_faces = 500000

        if isinstance(scene_or_mesh, trimesh.Scene):
            # Decimate individual geometries to stay under budget
            # while preserving per-geometry materials/textures
            total = sum(len(g.faces) for g in scene_or_mesh.geometry.values())
            if total > max_total_faces:
                ratio = max_total_faces / total
                for name, geom in scene_or_mesh.geometry.items():
                    target = max(100, int(len(geom.faces) * ratio))
                    if len(geom.faces) > target:
                        try:
                            scene_or_mesh.geometry[name] = geom.simplify_quadric_decimation(target)
                        except (ImportError, AttributeError):
                            pass  # keep original if simplification not available
            glb_data = scene_or_mesh.export(file_type="glb")
        else:
            mesh = scene_or_mesh
            if len(mesh.faces) > max_total_faces:
                try:
                    mesh = mesh.simplify_quadric_decimation(max_total_faces)
                except (ImportError, AttributeError):
                    pass  # keep original if simplification not available
            glb_data = mesh.export(file_type="glb")

        # Write to disk cache
        GLB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(glb_data)

        resp = Response(glb_data, mimetype="model/gltf-binary")
        resp.headers["Cache-Control"] = "public, max-age=604800, immutable"
        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500
