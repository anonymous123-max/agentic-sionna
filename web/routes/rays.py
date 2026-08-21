"""Ray tracing visualization routes."""

import math

from flask import Blueprint, jsonify, request

from routes.shared import _buildings_to_obstacles

rays_bp = Blueprint("rays", __name__)


@rays_bp.route("/api/rays/generate", methods=["POST"])
def generate_rays():
    """Generate synthetic ray paths from TX bouncing off walls."""
    data = request.json or {}
    tx_x = float(data.get("tx_x", 2.5))
    tx_y = float(data.get("tx_y", 2.0))
    tx_z = float(data.get("tx_z", 2.5))
    room_width = float(data.get("room_width", 5.0))
    room_length = float(data.get("room_length", 4.0))
    room_height = float(data.get("room_height", 2.7))
    num_rays = int(data.get("num_rays", 24))
    max_bounces = int(data.get("max_bounces", 3))

    # Parse wall polygons (list of polygons, each a list of [x,y] vertices)
    wall_polygons = data.get("wall_polygons", [])

    # Parse furniture for ray-furniture intersection
    furniture_boxes = []
    for f in data.get("furniture", []):
        fx, fy = float(f.get("x", 0)), float(f.get("y", 0))
        fw, fd, fh = float(f.get("width", 0)), float(f.get("depth", 0)), float(f.get("height", 0))
        if fw > 0 and fd > 0 and fh > 0:
            furniture_boxes.append({
                "xmin": fx - fw / 2, "xmax": fx + fw / 2,
                "ymin": fy - fd / 2, "ymax": fy + fd / 2,
                "zmin": 0.0, "zmax": fh,
            })

    # Append building footprints as AABB obstacles (for outdoor scenes)
    furniture_boxes.extend(_buildings_to_obstacles(data.get("buildings", [])))

    # Detect outdoor: buildings present or height > 10m suggests outdoor
    is_outdoor = len(data.get("buildings", [])) > 0 or room_height > 10

    rays = _generate_synthetic_rays(
        tx=(tx_x, tx_y, tx_z),
        room_dims=(room_width, room_length, room_height),
        num_rays=num_rays,
        max_bounces=max_bounces,
        furniture=furniture_boxes,
        is_outdoor=is_outdoor,
        wall_polygons=wall_polygons,
    )
    return jsonify({"rays": rays})


def _ray_aabb_intersect(pos, direction, box):
    """Ray-AABB intersection. Returns (t, normal) or (None, None).

    Uses the slab method to find ray entry point into an axis-aligned box.
    """
    tmin = 0.001
    tmax = 1e6
    normal = None

    for axis, (bmin_key, bmax_key) in enumerate(
        [("xmin", "xmax"), ("ymin", "ymax"), ("zmin", "zmax")]
    ):
        d = direction[axis]
        bmin = box[bmin_key]
        bmax = box[bmax_key]
        if abs(d) < 1e-9:
            # Ray parallel to slab
            if pos[axis] < bmin or pos[axis] > bmax:
                return None, None
            continue

        t1 = (bmin - pos[axis]) / d
        t2 = (bmax - pos[axis]) / d
        n1 = [0, 0, 0]
        n1[axis] = -1.0
        n2 = [0, 0, 0]
        n2[axis] = 1.0

        if t1 > t2:
            t1, t2 = t2, t1
            n1, n2 = n2, n1

        if t1 > tmin:
            tmin = t1
            normal = n1
        if t2 < tmax:
            tmax = t2

        if tmin > tmax:
            return None, None

    if tmin > 0.001 and normal is not None:
        return tmin, normal
    return None, None


def _ray_wall_segment_intersect(pos, direction, wall_seg, h):
    """Ray intersection with a vertical wall segment (2D line in XY, extruded to height h).

    wall_seg: ((ax, ay), (bx, by)) — two endpoints of wall edge.
    Returns (t, normal) or (None, None).
    """
    ox, oy = pos[0], pos[1]
    dx, dy = direction[0], direction[1]
    ax, ay = wall_seg[0]
    bx, by = wall_seg[1]

    # Wall direction vector
    wx, wy = bx - ax, by - ay
    # Solve: pos_xy + t * dir_xy = wall_start + s * wall_dir
    # Cross product denominator: dx * wy - dy * wx
    denom = dx * wy - dy * wx
    if abs(denom) < 1e-9:
        return None, None  # Parallel

    t = ((ax - ox) * wy - (ay - oy) * wx) / denom
    s = ((ax - ox) * dy - (ay - oy) * dx) / denom

    if t < 0.001 or s < -0.001 or s > 1.001:
        return None, None

    # Check Z at hit point
    hz = pos[2] + direction[2] * t
    if hz < -0.01 or hz > h + 0.01:
        return None, None

    # Wall inward normal (perpendicular, CW rotation of wall direction)
    seg_len = math.sqrt(wx * wx + wy * wy)
    if seg_len < 1e-6:
        return None, None
    nx, ny = wy / seg_len, -wx / seg_len
    # Orient normal to face the ray origin
    if nx * dx + ny * dy > 0:
        nx, ny = -nx, -ny

    return t, [nx, ny, 0.0]


def _build_wall_segments(wall_polygons):
    """Convert list of polygon vertex lists to list of wall segment pairs."""
    segments = []
    for poly in wall_polygons:
        if len(poly) < 3:
            continue
        for i in range(len(poly)):
            a = poly[i]
            b = poly[(i + 1) % len(poly)]
            segments.append(((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))))
    return segments


def _generate_synthetic_rays(
    tx: tuple,
    room_dims: tuple,
    num_rays: int,
    max_bounces: int,
    furniture: list = None,
    is_outdoor: bool = False,
    wall_polygons: list = None,
) -> list:
    """Cast rays from TX, reflect off walls and furniture, return path point lists.

    When wall_polygons are provided, rays intersect actual polygon wall segments
    instead of axis-aligned bounding planes.
    For outdoor scenes (is_outdoor=True), rays that hit the ceiling escape to
    the sky instead of reflecting — there is no ceiling in outdoor environments.
    """
    w, l, h = room_dims
    furniture = furniture or []
    wall_polygons = wall_polygons or []
    rays = []
    # Max travel distance per bounce scales with scene diagonal
    max_travel = math.sqrt(w * w + l * l + h * h) * 1.5

    use_polygon_walls = len(wall_polygons) > 0
    wall_segs = _build_wall_segments(wall_polygons) if use_polygon_walls else []

    for i in range(num_rays):
        azimuth = 2 * math.pi * i / num_rays
        # Vary elevation between -0.1 and -0.2 rad (~6°-12° below horizon)
        # so rays hit walls/floor rather than flying parallel to the ceiling.
        # Three elevation tiers via (i % 3) add visual depth to the ray fan.
        elevation = -0.1 - 0.3 * (i % 3) / 3

        dx = math.cos(azimuth) * math.cos(elevation)
        dy = math.sin(azimuth) * math.cos(elevation)
        dz = math.sin(elevation)

        points = [[tx[0], tx[1], tx[2]]]
        px, py, pz = tx
        bounces = 0

        for _ in range(max_bounces):
            # Find nearest wall/floor/ceiling intersection
            best_t = float("inf")
            best_normal = None
            hit_ceiling = False

            pos = [px, py, pz]
            direction = [dx, dy, dz]

            if use_polygon_walls:
                # Intersect against actual wall segments
                for seg in wall_segs:
                    t_hit, n_hit = _ray_wall_segment_intersect(pos, direction, seg, h)
                    if t_hit is not None and t_hit < best_t:
                        best_t = t_hit
                        best_normal = n_hit
                        hit_ceiling = False
            else:
                # Axis-aligned bounding box walls
                planes = [
                    (0, dx, 0.0),     # x = 0 (west wall)
                    (0, dx, w),       # x = w (east wall)
                    (1, dy, 0.0),     # y = 0 (south wall)
                    (1, dy, l),       # y = l (north wall)
                ]
                for axis, d_comp, boundary in planes:
                    if abs(d_comp) < 1e-9:
                        continue
                    t = (boundary - pos[axis]) / d_comp
                    if t > 0.001 and t < best_t:
                        hit = [pos[j] + direction[j] * t for j in range(3)]
                        if (
                            -0.01 <= hit[0] <= w + 0.01
                            and -0.01 <= hit[1] <= l + 0.01
                            and -0.01 <= hit[2] <= h + 0.01
                        ):
                            best_t = t
                            normal = [0, 0, 0]
                            normal[axis] = -1.0 if d_comp > 0 else 1.0
                            best_normal = normal

            # Floor and ceiling planes (always axis-aligned)
            for axis, d_comp, boundary, is_ceil in [
                (2, dz, 0.0, False),
                (2, dz, h, True),
            ]:
                if abs(d_comp) < 1e-9:
                    continue
                t = (boundary - pos[axis]) / d_comp
                if t > 0.001 and t < best_t:
                    best_t = t
                    normal = [0, 0, 0]
                    normal[axis] = -1.0 if d_comp > 0 else 1.0
                    best_normal = normal
                    hit_ceiling = is_ceil

            # Check furniture AABB intersections
            for box in furniture:
                t_hit, n_hit = _ray_aabb_intersect(pos, direction, box)
                if t_hit is not None and t_hit < best_t:
                    best_t = t_hit
                    best_normal = n_hit
                    hit_ceiling = False  # Hit an object, not the ceiling

            if best_normal is None or best_t > max_travel:
                break

            # Move to hit point
            px += dx * best_t
            py += dy * best_t
            pz += dz * best_t
            pz = max(0, min(h, pz))

            points.append([round(px, 4), round(py, 4), round(pz, 4)])
            bounces += 1

            # Outdoor: ray escapes to sky on ceiling hit — stop tracing
            if is_outdoor and hit_ceiling:
                break

            # Reflect direction: d = d - 2*(d.n)*n
            dot = dx * best_normal[0] + dy * best_normal[1] + dz * best_normal[2]
            dx -= 2 * dot * best_normal[0]
            dy -= 2 * dot * best_normal[1]
            dz -= 2 * dot * best_normal[2]

        if len(points) > 1:
            rays.append({"points": points, "bounces": bounces})

    return rays
