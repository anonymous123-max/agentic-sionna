"""Mesh segmentation API routes.

Provides endpoints for triggering segmentation of EFM3D PLY meshes,
polling job progress, and serving segmentation outputs (metadata, GLB,
XML, individual OBJ segments).
"""

import json
import logging
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from routes.shared import (
    _create_job,
    _update_job,
    _finish_job,
    _fail_job,
    is_job_cancelled,
)

logger = logging.getLogger(__name__)

segment_bp = Blueprint("segmentation", __name__)

DATA_DIR = Path("data/scenes")
OUTPUT_DIR = Path("outputs/segmented")


@segment_bp.route("/api/segment/<scene_id>", methods=["POST"])
def start_segmentation(scene_id):
    """Start a segmentation job for an EFM3D scene.

    Accepts JSON: { use_ase?: bool (default true) }

    Fast path: if outputs/segmented/{scene_id}/segmentation.json exists,
    returns cached metadata immediately.

    Slow path: starts a background thread and returns { job_id }.
    """
    scene_dir = DATA_DIR / scene_id
    ply_path = scene_dir / "scene_mesh.ply"

    if not ply_path.exists():
        return jsonify({"error": f"No PLY mesh for scene {scene_id}"}), 404

    # Check cache — require "rooms" key (multi-room format v2)
    meta_path = OUTPUT_DIR / scene_id / "segmentation.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                metadata = json.load(f)
            if "rooms" in metadata:
                return jsonify(metadata)
            # Old format without rooms — re-segment
            logger.info("Scene %s: stale cache (no rooms), re-segmenting", scene_id)
        except Exception:
            pass

    data = request.get_json(silent=True) or {}
    use_ase = data.get("use_ase", True)

    ase_path = None
    if use_ase:
        ase_file = scene_dir / "ase_scene_language.txt"
        if ase_file.exists():
            ase_path = ase_file

    job_id = _create_job()
    output_dir = OUTPUT_DIR / scene_id

    def work():
        try:
            from src.mesh.segmentation import segment_mesh
            from src.mesh.exporter import export_segments

            def on_progress(pct, msg):
                if is_job_cancelled(job_id):
                    raise RuntimeError("Cancelled by user")
                _update_job(job_id, pct, msg)

            _update_job(job_id, 5, "Starting segmentation...")
            result, mesh = segment_mesh(
                ply_path, ase_path=ase_path, on_progress=on_progress,
            )

            _update_job(job_id, 85, "Exporting segments...")
            metadata = export_segments(
                result, mesh, output_dir,
                on_progress=lambda pct, msg: on_progress(85 + pct * 15 // 100, msg),
            )

            _finish_job(job_id, metadata)
        except Exception as e:
            logger.exception("Segmentation failed for scene %s", scene_id)
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@segment_bp.route("/api/segment/<scene_id>/status")
def segmentation_status(scene_id):
    """Check if segmentation results exist for a scene."""
    meta_path = OUTPUT_DIR / scene_id / "segmentation.json"
    if meta_path.exists():
        return jsonify({"segmented": True})
    return jsonify({"segmented": False})


@segment_bp.route("/api/segment/<scene_id>/result")
def segmentation_result(scene_id):
    """Get segmentation metadata for a scene."""
    meta_path = OUTPUT_DIR / scene_id / "segmentation.json"
    if not meta_path.exists():
        return jsonify({"error": "Not segmented yet"}), 404

    with open(meta_path) as f:
        metadata = json.load(f)
    return jsonify(metadata)


@segment_bp.route("/api/segment/<scene_id>/mesh")
def segmentation_mesh(scene_id):
    """Serve the visualization GLB for a segmented scene."""
    glb_path = OUTPUT_DIR / scene_id / "segmented.glb"
    if not glb_path.exists():
        return jsonify({"error": "No segmented mesh available"}), 404
    return send_file(str(glb_path.resolve()), mimetype="model/gltf-binary")


@segment_bp.route("/api/segment/<scene_id>/xml")
def segmentation_xml(scene_id):
    """Serve the Sionna XML scene for a segmented scene."""
    xml_path = OUTPUT_DIR / scene_id / "scene.xml"
    if not xml_path.exists():
        return jsonify({"error": "No XML scene available"}), 404
    return send_file(str(xml_path.resolve()), mimetype="application/xml")


@segment_bp.route("/api/segment/<scene_id>/segment/<int:idx>")
def segmentation_segment(scene_id, idx):
    """Serve an individual segment OBJ file by index."""
    meta_path = OUTPUT_DIR / scene_id / "segmentation.json"
    if not meta_path.exists():
        return jsonify({"error": "Not segmented yet"}), 404

    with open(meta_path) as f:
        metadata = json.load(f)

    segments = metadata.get("segments", [])
    if idx < 0 or idx >= len(segments):
        return jsonify({"error": f"Segment index {idx} out of range"}), 404

    filename = segments[idx]["filename"]
    obj_path = OUTPUT_DIR / scene_id / "segments" / filename
    if not obj_path.exists():
        return jsonify({"error": f"Segment file not found: {filename}"}), 404

    return send_file(str(obj_path.resolve()), mimetype="text/plain")
