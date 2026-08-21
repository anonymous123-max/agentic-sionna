"""Scene creation routes: indoor, outdoor, OSM fetch."""

import json
import logging
import math
import threading
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request

from src.models.room import BoundingBox as RoomBBox, Door, FurnitureItem, Position, Room
from src.models.outdoor import Building, GroundPlane, OutdoorScene
from src.models.scene import Scene
from src.optimizer.layout import FurnitureSpec, optimize_layout, BoundingBox
from src.exporters.xml import XMLExporter

from routes.shared import (
    OUTPUTS_DIR,
    GENERIC_FURNITURE_DIMS,
    _create_job,
    _scene_name,
    _save_scene,
    _update_job,
    _finish_job,
    _fail_job,
    _get_catalog,
    _resolve_model_path,
    _room_furniture_to_list,
    _build_indoor_result,
    _build_indoor_metadata,
    is_job_cancelled,
)

logger = logging.getLogger(__name__)

creation_bp = Blueprint("creation", __name__)


# ─────────────────────────────────────────────────────────
# Indoor scene creation
# ─────────────────────────────────────────────────────────

@creation_bp.route("/api/scene/indoor/create", methods=["POST"])
def create_indoor_scene():
    """Create indoor room with furniture and export XML."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Parsing parameters...")
            room_width = float(data.get("room_width", 5.0))
            room_length = float(data.get("room_length", 4.0))
            floor_polygon = data.get("floor_polygon")

            door_data = data.get("door", {"wall": "south", "position": 2.0, "width": 0.9})
            doors = [Door(**door_data)]

            _update_job(job_id, 30, "Optimizing layout...")
            room = optimize_layout(
                room_width=room_width,
                room_length=room_length,
                doors=doors,
                floor_polygon=floor_polygon,
            )

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 70, "Exporting XML...")
            scene = Scene(room=room)
            name = _scene_name("Room", f"{room_width:.0f}x{room_length:.0f}", job_id)
            _save_scene(scene, name, _build_indoor_metadata(room, name))

            _update_job(job_id, 90, "Preparing result...")
            xml_path = OUTPUTS_DIR / name / "scene.xml"
            result = _build_indoor_result(room, name, xml_path)
            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("Indoor scene creation failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@creation_bp.route("/api/scene/indoor/create-with-furniture", methods=["POST"])
def create_indoor_with_furniture():
    """Create indoor room with specified furniture, optimize, and export XML."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Parsing parameters...")
            room_width = float(data.get("room_width", 5.0))
            room_length = float(data.get("room_length", 4.0))
            floor_polygon = data.get("floor_polygon")

            door_data = data.get("door", {"wall": "south", "position": 2.0, "width": 0.9})
            doors = [Door(**door_data)]

            catalog = _get_catalog()
            specs = []

            # Re-include existing furniture so they get re-optimized alongside new items
            for ef in data.get("existing_furniture", []):
                dims = BoundingBox(
                    width=float(ef.get("width", 0.5)),
                    depth=float(ef.get("depth", 0.5)),
                    height=float(ef.get("height", 0.5)),
                )
                ef_model_id = ef.get("model_id", str(uuid.uuid4()))
                ef_model_path = ef.get("model_path", "")
                ef_model_file = ef.get("model_file", "")
                if ef_model_id and not ef_model_path and not ef_model_file:
                    ef_model_path, ef_model_file = _resolve_model_path(ef_model_id, catalog)
                specs.append(FurnitureSpec(
                    category=ef.get("category", "unknown"),
                    model_id=ef_model_id,
                    model_path=ef_model_path,
                    model_file=ef_model_file,
                    dimensions=dims,
                ))

            # Add new furniture from AI action
            for item in data.get("furniture", []):
                cat = item["category"]
                qty = int(item.get("quantity", 1))
                requested_model_id = item.get("model_id")
                for _ in range(qty):
                    model_id = str(uuid.uuid4())
                    model_path = ""
                    model_file = ""
                    dims = GENERIC_FURNITURE_DIMS.get(
                        cat.lower(), BoundingBox(width=0.5, depth=0.5, height=0.5)
                    )
                    if catalog is not None:
                        try:
                            if requested_model_id:
                                model_id = requested_model_id
                            else:
                                model = catalog.get_random_model(cat)
                                model_id = model["model_id"]
                            dims = catalog.get_dimensions(model_id)
                            model_path, model_file = _resolve_model_path(model_id, catalog)
                        except (ValueError, KeyError):
                            logger.debug("Catalog lookup failed for %s, using generic dims", cat)
                    specs.append(FurnitureSpec(
                        category=cat,
                        model_id=model_id,
                        model_path=model_path,
                        model_file=model_file,
                        dimensions=dims,
                    ))

            # Cap furniture count based on room area to prevent overcrowding
            room_area = room_width * room_length
            max_items = max(3, int(room_area / 3))
            if len(specs) > max_items:
                logger.info("Trimming furniture from %d to %d (room area %.0f m²)",
                            len(specs), max_items, room_area)
                specs = specs[:max_items]

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 30, f"Optimizing layout for {len(specs)} items...")
            room = optimize_layout(
                room_width=room_width,
                room_length=room_length,
                doors=doors,
                furniture_specs=specs,
                floor_polygon=floor_polygon,
            )

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 70, "Exporting XML...")
            scene = Scene(room=room)
            detail = data.get("name", f"{room_width:.0f}x{room_length:.0f}")
            name = _scene_name("Room", detail, job_id)
            _save_scene(scene, name, _build_indoor_metadata(room, name))

            _update_job(job_id, 90, "Preparing result...")
            xml_path = OUTPUTS_DIR / name / "scene.xml"
            result = _build_indoor_result(room, name, xml_path)
            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("Indoor scene creation with furniture failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@creation_bp.route("/api/scene/indoor/resize", methods=["POST"])
def resize_indoor_scene():
    """Resize room dimensions in-place, clamping existing furniture positions."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Parsing parameters...")
            room_width = float(data.get("room_width", 5.0))
            room_length = float(data.get("room_length", 4.0))
            room_height = float(data.get("room_height", 2.7))
            floor_polygon = data.get("floor_polygon")

            door_data = data.get("door", {"wall": "south", "position": min(2.0, room_width / 2), "width": 0.9})
            doors = [Door(**door_data)]

            catalog = _get_catalog()
            clamped_items = []
            for f in data.get("furniture", []):
                fw = float(f.get("width", 0.5))
                fd = float(f.get("depth", 0.5))
                fh = float(f.get("height", 0.5))
                fx = float(f.get("x", 0))
                fy = float(f.get("y", 0))

                # Clamp to keep furniture inside new room bounds
                fx = max(fw / 2, min(fx, room_width - fw / 2))
                fy = max(fd / 2, min(fy, room_length - fd / 2))

                f_model_id = f.get("model_id", "")
                f_model_path = f.get("model_path", "")
                f_model_file = f.get("model_file", "")
                if f_model_id and not f_model_path and not f_model_file:
                    f_model_path, f_model_file = _resolve_model_path(f_model_id, catalog)

                clamped_items.append(FurnitureItem(
                    id=f.get("id", str(uuid.uuid4())),
                    category=f.get("category", "unknown"),
                    model_id=f_model_id,
                    model_path=f_model_path,
                    model_file=f_model_file,
                    position=Position(
                        x=fx, y=fy,
                        theta=math.radians(float(f.get("theta", 0))),
                    ),
                    dimensions=RoomBBox(width=fw, depth=fd, height=fh),
                ))

            _update_job(job_id, 50, "Building resized room...")
            room = Room(
                width=room_width,
                length=room_length,
                height=room_height,
                floor_polygon=[tuple(p) for p in floor_polygon] if floor_polygon else None,
                doors=doors,
                furniture=clamped_items,
            )

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 70, "Exporting XML...")
            scene = Scene(room=room)
            name = _scene_name("Room", f"{room_width:.0f}x{room_length:.0f}", job_id)
            _save_scene(scene, name, _build_indoor_metadata(room, name))

            _update_job(job_id, 90, "Preparing result...")
            xml_path = OUTPUTS_DIR / name / "scene.xml"
            result = _build_indoor_result(room, name, xml_path)
            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("Indoor scene resize failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


# ─────────────────────────────────────────────────────────
# Outdoor scene creation
# ─────────────────────────────────────────────────────────

def _buildings_to_list(buildings: list) -> list:
    """Convert Building objects to JSON-serializable dicts."""
    return [
        {"id": b.id, "footprint": b.footprint, "height": b.height,
         "material": b.material, "bounds": list(b.bounds),
         "name": getattr(b, 'name', None)}
        for b in buildings
    ]


@creation_bp.route("/api/scene/outdoor/create", methods=["POST"])
def create_outdoor_scene():
    """Create outdoor scene from bbox or predefined buildings."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 10, "Building scene...")

            width = float(data.get("width", 200.0))
            length = float(data.get("length", 200.0))
            ground_material = data.get("ground_material", "wet_ground")

            buildings = []
            for i, b in enumerate(data.get("buildings", [])):
                buildings.append(Building(
                    id=b.get("id", f"bldg_{i}"),
                    footprint=[tuple(p) for p in b["footprint"]],
                    height=float(b.get("height", 12.0)),
                    material=b.get("material", "concrete"),
                ))

            _update_job(job_id, 40, "Creating scene model...")
            outdoor = OutdoorScene(
                width=width,
                length=length,
                ground=GroundPlane(width=width, length=length, material=ground_material),
                buildings=buildings,
            )

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 60, "Exporting XML...")
            scene = Scene(outdoor=outdoor)
            detail = data.get("name", f"{width:.0f}x{length:.0f}")
            name = _scene_name("Outdoor", detail, job_id)
            buildings_list = _buildings_to_list(buildings)

            _save_scene(scene, name, {
                "type": "outdoor",
                "name": name,
                "width": width,
                "length": length,
                "ground_material": ground_material,
                "buildings": buildings_list,
            })

            _update_job(job_id, 90, "Done")
            result = {
                "scene_id": name,
                "xml_path": str(OUTPUTS_DIR / name / "scene.xml"),
                "width": width,
                "length": length,
                "num_buildings": len(buildings),
                "buildings": buildings_list,
            }
            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("Outdoor scene creation failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})


@creation_bp.route("/api/building/<building_id>/enter", methods=["POST"])
def enter_building(building_id):
    """Building entry has been disabled."""
    return jsonify({"error": "Building entry is no longer available"}), 410


# ─────────────────────────────────────────────────────────
# OpenStreetMap fetch
# ─────────────────────────────────────────────────────────

# Maximum OSM download area: 0.25 km² (~500m x 500m).
# Dense areas like Manhattan can have 16,000+ buildings in a few km².
MAX_OSM_AREA_M2 = 250_000


@creation_bp.route("/api/scene/osm/fetch", methods=["POST"])
def fetch_osm_scene():
    """Fetch outdoor scene from OpenStreetMap by bbox or location name."""
    data = request.json or {}
    job_id = _create_job()

    def work():
        try:
            _update_job(job_id, 5, "Importing OSM modules...")
            from src.osm.config import OSMConfig

            osm_config = OSMConfig(
                include_buildings=True,
                include_roads=data.get("include_roads", True),
                include_trees=False,  # Trees rarely available, skip for speed
            )

            # Support either bbox (north/south/east/west) or lat/lon + radius
            if all(k in data for k in ("north", "south", "east", "west")):
                north = float(data["north"])
                south = float(data["south"])
                east = float(data["east"])
                west = float(data["west"])
                location_name = data.get("name", "")
            elif "lat" in data and "lon" in data:
                lat = float(data["lat"])
                lon = float(data["lon"])
                radius_m = float(data.get("radius", 300))
                # Approximate degree offsets from meters
                dlat = radius_m / 111320.0
                dlon = radius_m / (111320.0 * math.cos(math.radians(lat)))
                north = lat + dlat
                south = lat - dlat
                east = lon + dlon
                west = lon - dlon
                location_name = data.get("name", f"{lat:.4f}_{lon:.4f}")
            else:
                _fail_job(job_id, "Provide north/south/east/west or lat/lon")
                return

            # Area limit check
            mid_lat = (north + south) / 2
            width_m = abs(east - west) * 111320.0 * math.cos(math.radians(mid_lat))
            height_m = abs(north - south) * 111320.0
            area_m2 = width_m * height_m
            if area_m2 > MAX_OSM_AREA_M2 and not data.get("force"):
                _fail_job(
                    job_id,
                    f"Requested area too large: {area_m2 / 1e6:.2f} km² "
                    f"({width_m:.0f}m x {height_m:.0f}m). "
                    f"Max is {MAX_OSM_AREA_M2 / 1e6:.2f} km² (~500m x 500m). "
                    f"Use a smaller bounding box or radius.",
                )
                return

            if is_job_cancelled(job_id):
                return

            from src.osm.fetcher import fetch_buildings, fetch_roads
            from src.osm.converter import convert_to_outdoor_scene

            bbox = (west, south, east, north)

            _update_job(job_id, 10, f"Downloading buildings for {location_name or 'bbox'}...")
            buildings_gdf = fetch_buildings(bbox=bbox) if osm_config.include_buildings else None
            n_buildings = len(buildings_gdf) if buildings_gdf is not None and not buildings_gdf.empty else 0

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 35, f"Downloaded {n_buildings} buildings. Downloading roads...")
            roads_gdf = fetch_roads(bbox=bbox, network_type=osm_config.road_network_type) if osm_config.include_roads else None
            n_roads = len(roads_gdf) if roads_gdf is not None and not roads_gdf.empty else 0

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 50, f"Downloaded {n_roads} roads. Converting geometry...")
            outdoor = convert_to_outdoor_scene(buildings_gdf, roads_gdf, None, osm_config)

            if is_job_cancelled(job_id):
                return

            _update_job(job_id, 70, f"Converted {len(outdoor.buildings)} buildings. Exporting XML...")
            scene = Scene(outdoor=outdoor)
            name = _scene_name("OSM", location_name or f"{outdoor.width:.0f}x{outdoor.length:.0f}", job_id)

            buildings_list = _buildings_to_list(outdoor.buildings)
            roads_list = []
            if hasattr(outdoor, 'roads') and outdoor.roads:
                roads_list = [
                    {"id": r.id, "centerline": r.centerline, "width": r.width,
                     "material": r.material}
                    for r in outdoor.roads
                ]

            _save_scene(scene, name, {
                "type": "outdoor",
                "source": "osm",
                "name": name,
                "location": location_name,
                "width": outdoor.width,
                "length": outdoor.length,
                "ground_material": "wet_ground",
                "buildings": buildings_list,
                "roads": roads_list,
            })

            _update_job(job_id, 90, "Done")
            result = {
                "scene_id": name,
                "xml_path": str(OUTPUTS_DIR / name / "scene.xml"),
                "width": outdoor.width,
                "length": outdoor.length,
                "num_buildings": len(outdoor.buildings),
                "buildings": buildings_list,
                "roads": roads_list,
            }
            _finish_job(job_id, result)
        except Exception as e:
            logger.exception("OSM fetch failed")
            _fail_job(job_id, str(e))

    threading.Thread(target=work, daemon=True).start()
    return jsonify({"job_id": job_id})
