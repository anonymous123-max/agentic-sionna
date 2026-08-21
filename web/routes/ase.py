"""ASE (Aria Synthetic Environments) scene routes.

Provides endpoints for loading and listing ASE apartment layouts,
parsing them into Room models with ABO-matched furniture, and
running the full Aria→Sionna reconstruction pipeline.
"""

import json
import logging
import math
import threading
from collections import OrderedDict
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from routes.shared import (
    _get_catalog,
    _create_job,
    _update_job,
    _finish_job,
    _fail_job,
    is_job_cancelled,
)

logger = logging.getLogger(__name__)

ase_bp = Blueprint("ase", __name__)

ASE_SCENES_DIR = Path("data/ase-scenes")
ASE_MESHES_DIR = Path("data/ase-meshes")
ASE_SIONNA_DIR = Path("outputs/ase-sionna")

_ASE_CACHE_MAX = 16
_ase_cache = OrderedDict()
_ase_cache_lock = threading.Lock()


def _parse_ase_scene(scene_file_str: str):
    """Parse and cache an ASE scene (thread-safe, LRU eviction)."""
    cache_key = scene_file_str
    with _ase_cache_lock:
        if cache_key in _ase_cache:
            _ase_cache.move_to_end(cache_key)
            return _ase_cache[cache_key]

    # Parse outside the lock (expensive)
    from src.catalog.ase_parser import ASESceneParser

    catalog = _get_catalog()
    parser = ASESceneParser(abo_catalog=catalog)
    result = parser.parse_file_rooms(Path(scene_file_str))

    with _ase_cache_lock:
        _ase_cache[cache_key] = result
        _ase_cache.move_to_end(cache_key)
        while len(_ase_cache) > _ASE_CACHE_MAX:
            _ase_cache.popitem(last=False)

    return result


@ase_bp.route("/api/scene/ase/clear-cache", methods=["POST"])
def ase_clear_cache():
    """Flush the in-memory ASE scene parser cache.

    Call this after code changes to the parser so stale results
    (e.g. with wrong catalog models) are not reused.
    """
    with _ase_cache_lock:
        count = len(_ase_cache)
        _ase_cache.clear()
    logger.info("Cleared ASE scene cache (%d entries)", count)
    return jsonify({"cleared": count})


@ase_bp.route("/api/scene/ase/list")
def ase_list_scenes():
    """List downloaded ASE scenes in data/ase-scenes/.

    Returns JSON with list of scene IDs and file paths.
    """
    if not ASE_SCENES_DIR.exists():
        return jsonify({"scenes": [], "message": "No ASE scenes directory found"})

    scenes = []
    for scene_file in sorted(ASE_SCENES_DIR.rglob("ase_scene_language.txt")):
        scene_id = scene_file.parent.name
        scenes.append({
            "scene_id": scene_id,
            "path": str(scene_file),
        })

    return jsonify({"scenes": scenes, "total": len(scenes)})


@ase_bp.route("/api/scene/ase/load", methods=["POST"])
def ase_load_scene():
    """Load and parse an ASE scene into individual rooms.

    Accepts JSON with either:
    - scene_id: ID of a scene in data/ase-scenes/<scene_id>/
    - file_path: Direct path to ase_scene_language.txt

    Optionally:
    - room_index: Index of specific room to load (default: largest)

    Returns parsed rooms with furniture placed by the optimizer.
    """
    data = request.get_json(silent=True) or {}
    scene_id = data.get("scene_id", "")
    file_path = data.get("file_path", "")
    room_index = data.get("room_index", None)

    # Resolve scene file path
    if scene_id:
        scene_file = ASE_SCENES_DIR / scene_id / "ase_scene_language.txt"
    elif file_path:
        scene_file = Path(file_path)
    else:
        return jsonify({"error": "Provide scene_id or file_path"}), 400

    if not scene_file.exists():
        return jsonify({"error": f"Scene file not found: {scene_file}"}), 404

    # Check for pipeline data
    has_pipeline = False
    pipeline_meta_path = ASE_SIONNA_DIR / scene_id / "metadata.json" if scene_id else None
    if pipeline_meta_path and pipeline_meta_path.exists():
        has_pipeline = True

    try:
        room_list = _parse_ase_scene(str(scene_file))

        if not room_list:
            return jsonify({"error": "No rooms detected in scene"}), 404

        # Normalize offsets so apartment min is at origin
        all_offsets = [(info.x_offset, info.y_offset) for _, info in room_list]
        global_min_x = min(o[0] for o in all_offsets) if all_offsets else 0.0
        global_min_y = min(o[1] for o in all_offsets) if all_offsets else 0.0

        # Build room summary + full data for every room
        rooms_info = []
        all_rooms = []
        for room, info in room_list:
            rx = info.x_offset - global_min_x
            ry = info.y_offset - global_min_y

            rooms_info.append({
                "index": info.index,
                "label": info.label,
                "width": info.width,
                "length": info.length,
                "height": info.height,
                "area": info.area,
                "wall_count": info.wall_count,
                "furniture_count": len(room.furniture),
                "door_count": len(room.doors),
                "window_count": len(room.windows),
            })

            furniture_list = []
            for f in room.furniture:
                furniture_list.append({
                    "id": f.id,
                    "category": f.category,
                    "model_id": f.model_id,
                    "model_file": getattr(f, "model_file", ""),
                    "x": f.position.x,
                    "y": f.position.y,
                    "theta": math.degrees(f.position.theta),
                    "width": f.dimensions.width,
                    "depth": f.dimensions.depth,
                    "height": f.dimensions.height,
                })

            all_rooms.append({
                "index": info.index,
                "label": info.label,
                "width": info.width,
                "length": info.length,
                "height": info.height,
                "x_offset": round(rx, 3),
                "y_offset": round(ry, 3),
                "floor_polygon": room.floor_polygon,
                "doors": [d.model_dump() for d in room.doors],
                "windows": [w.model_dump() for w in room.windows],
                "furniture": furniture_list,
            })

        # Select primary room for backward-compat fields
        if room_index is not None:
            idx = int(room_index)
            matches = [(r, i) for r, i in room_list if i.index == idx]
            if not matches:
                return jsonify({"error": f"Room index {idx} not found"}), 404
            selected_room, selected_info = matches[0]
        else:
            room_list_sorted = sorted(room_list, key=lambda r: r[1].area, reverse=True)
            selected_room, selected_info = room_list_sorted[0]

        result = {
            "width": selected_room.width,
            "length": selected_room.length,
            "height": selected_room.height,
            "floor_polygon": selected_room.floor_polygon,
            "room_index": selected_info.index,
            "room_label": selected_info.label,
            "rooms": rooms_info,
            "all_rooms": all_rooms,
        }

        # Enrich with pipeline data if cached
        if has_pipeline:
            try:
                with open(pipeline_meta_path) as fh:
                    meta = json.load(fh)
                instances = meta.get("instances", [])

                result["pipeline_instances"] = instances
                result["pipeline_origin"] = [global_min_x, global_min_y]
                result["has_cloud"] = meta.get("has_cloud", False)
                result["cloud_num_points"] = meta.get("cloud_num_points", 0)
                # Also check if cloud file exists even if metadata doesn't flag it
                cloud_file = ASE_SIONNA_DIR / scene_id / "scene_cloud.bin"
                if cloud_file.exists() and not result["has_cloud"]:
                    result["has_cloud"] = True
                # Scene mesh (reconstructed from depth maps)
                result["has_mesh"] = meta.get("has_mesh", False)
                mesh_file = ASE_SIONNA_DIR / scene_id / "scene_mesh.glb"
                if mesh_file.exists() and not result["has_mesh"]:
                    result["has_mesh"] = True
                xml_file = ASE_SIONNA_DIR / scene_id / "scene.xml"
                if xml_file.exists():
                    result["xml_path"] = str(xml_file)
                abo_matched = sum(1 for i in instances if i.get("abo_model_id"))
                total_furn = sum(len(r.get("furniture", []))
                                 for r in all_rooms)
                logger.info(
                    "ASE scene %s: %d pipeline instances (%d ABO-matched), "
                    "%d parser furniture, origin=[%.2f, %.2f]",
                    scene_id, len(instances), abo_matched, total_furn,
                    global_min_x, global_min_y)
            except Exception:
                logger.warning(
                    "Failed to load pipeline metadata for %s",
                    scene_id, exc_info=True)
        else:
            total_furn = sum(len(r.get("furniture", []))
                             for r in all_rooms)
            logger.info(
                "ASE scene %s: no pipeline data, %d optimizer furniture",
                scene_id, total_furn)

        # Check for EFM3D PLY mesh (data/scenes/<id>/scene_mesh.ply)
        ply_path = Path("data/scenes") / scene_id / "scene_mesh.ply"
        result["has_ply_mesh"] = ply_path.exists()

        # Check if already segmented
        seg_meta = Path("outputs/segmented") / scene_id / "segmentation.json"
        result["segmented"] = seg_meta.exists()

        return jsonify(result)

    except Exception as exc:
        logger.exception("Failed to parse ASE scene: %s", exc)
        return jsonify({"error": str(exc)}), 500


@ase_bp.route("/api/scene/ase/mesh/<mesh_id>")
def ase_get_mesh(mesh_id):
    """Serve a simplified GLB mesh for an ASE evaluation scene.

    If only the raw PLY exists, converts and caches the simplified GLB.
    Returns 404 if no mesh is available for this ID.
    """
    mesh_dir = ASE_MESHES_DIR / mesh_id
    glb_path = mesh_dir / "scene_simplified.glb"

    if glb_path.exists():
        return send_file(glb_path, mimetype="model/gltf-binary")

    # Check for raw PLY to convert
    ply_path = mesh_dir / f"scene_ply_{mesh_id}.ply"
    if not ply_path.exists():
        return jsonify({"error": f"No mesh found for ID {mesh_id}"}), 404

    try:
        from src.catalog.mesh_utils import simplify_ply_to_glb
        simplify_ply_to_glb(ply_path, glb_path)
        return send_file(glb_path, mimetype="model/gltf-binary")
    except Exception as exc:
        logger.exception("Failed to convert mesh %s: %s", mesh_id, exc)
        return jsonify({"error": str(exc)}), 500


@ase_bp.route("/api/scene/ase/meshes")
def ase_list_meshes():
    """List available ASE evaluation meshes."""
    if not ASE_MESHES_DIR.exists():
        return jsonify({"meshes": [], "total": 0})

    meshes = []
    for d in sorted(ASE_MESHES_DIR.iterdir()):
        if not d.is_dir():
            continue
        has_glb = (d / "scene_simplified.glb").exists()
        has_ply = any(d.glob("scene_ply_*.ply"))
        if has_glb or has_ply:
            meshes.append({
                "mesh_id": d.name,
                "has_glb": has_glb,
                "has_ply": has_ply,
            })

    return jsonify({"meshes": meshes, "total": len(meshes)})


# ─────────────────────────────────────────────────────────
# Aria → Sionna full reconstruction pipeline
# ─────────────────────────────────────────────────────────


@ase_bp.route("/api/scene/ase/sionna", methods=["POST"])
def ase_sionna_pipeline():
    """Run the full Aria→Sionna pipeline for a scene.

    Accepts JSON: { scene_id, frame_step? }

    Fast path: if outputs/ase-sionna/{scene_id}/metadata.json exists,
    returns cached metadata immediately.

    Slow path: creates SSE job, runs pipeline in background thread,
    returns { job_id }.
    """
    data = request.get_json(silent=True) or {}
    scene_id = str(data.get("scene_id", "")).strip()
    if not scene_id:
        return jsonify({"error": "scene_id required"}), 400

    frame_step = int(data.get("frame_step", 10))

    # Fast path: cached result
    meta_path = ASE_SIONNA_DIR / scene_id / "metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            return jsonify(metadata)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read cached ASE pipeline metadata: %s", e)

    # Slow path: background job
    job_id = _create_job()

    def work():
        try:
            from scripts.aria_to_sionna import run_pipeline

            def on_progress(pct, msg):
                if is_job_cancelled(job_id):
                    raise RuntimeError("Cancelled by user")
                _update_job(job_id, pct, msg)

            metadata = run_pipeline(
                scene_id=scene_id,
                frame_step=frame_step,
                on_progress=on_progress,
            )
            _finish_job(job_id, metadata)
        except Exception as e:
            logger.exception("ASE Sionna pipeline failed for scene %s", scene_id)
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@ase_bp.route("/api/scene/ase/sionna/mesh/<scene_id>/<filename>")
def ase_sionna_mesh(scene_id, filename):
    """Serve a reconstructed furniture mesh as GLB.

    OBJ files are auto-converted to GLB (cached alongside the .obj).
    """
    meshes_dir = ASE_SIONNA_DIR / scene_id / "meshes"

    # If requesting .glb directly, check if it exists
    glb_path = meshes_dir / filename
    if glb_path.exists() and filename.endswith(".glb"):
        return send_file(str(glb_path), mimetype="model/gltf-binary")

    # If .obj file, convert to .glb (cached)
    obj_name = filename.replace(".glb", ".obj") if filename.endswith(".glb") else filename
    obj_path = meshes_dir / obj_name
    if not obj_path.exists():
        return jsonify({"error": f"Mesh not found: {filename}"}), 404

    glb_path = obj_path.with_suffix(".glb")
    if not glb_path.exists():
        try:
            import trimesh
            mesh = trimesh.load(str(obj_path), force="mesh")
            mesh.export(str(glb_path), file_type="glb")
        except Exception as exc:
            logger.exception("OBJ→GLB conversion failed: %s", exc)
            return jsonify({"error": str(exc)}), 500

    return send_file(str(glb_path), mimetype="model/gltf-binary")


@ase_bp.route("/api/scene/ase/sionna/cloud/<scene_id>")
def ase_sionna_cloud(scene_id):
    """Serve the scene point cloud binary for frontend rendering."""
    cloud_path = ASE_SIONNA_DIR / scene_id / "scene_cloud.bin"
    if not cloud_path.exists():
        return jsonify({"error": "No point cloud for this scene"}), 404
    return send_file(str(cloud_path), mimetype="application/octet-stream")


@ase_bp.route("/api/scene/ase/sionna/scene-mesh/<scene_id>")
def ase_sionna_scene_mesh(scene_id):
    """Serve the reconstructed full scene mesh as GLB."""
    glb_path = ASE_SIONNA_DIR / scene_id / "scene_mesh.glb"
    if not glb_path.exists():
        return jsonify({"error": "No scene mesh for this scene"}), 404
    return send_file(str(glb_path), mimetype="model/gltf-binary")


@ase_bp.route("/api/scene/ase/sionna/bundle/<scene_id>")
def ase_sionna_bundle(scene_id):
    """Serve a pre-built scene bundle for download.

    Returns the bundle.tar.gz for the given scene, or 404 if not built yet.
    Use --bundle flag with aria_to_sionna.py to create bundles.
    """
    bundle_path = ASE_SIONNA_DIR / scene_id / "bundle.tar.gz"
    if not bundle_path.exists():
        return jsonify({"error": f"No bundle for scene {scene_id}. "
                        "Run pipeline with --bundle to create one."}), 404

    return send_file(
        str(bundle_path),
        mimetype="application/gzip",
        as_attachment=True,
        download_name=f"ase-scene-{scene_id}-bundle.tar.gz",
    )
