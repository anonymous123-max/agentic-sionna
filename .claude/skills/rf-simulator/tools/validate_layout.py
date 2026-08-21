#!/usr/bin/env python3
"""Validate (and optionally fix) furniture placement in a scene_state.json.

Usage:
    python3 validate_layout.py scene_state.json           # report only, exit 1 if invalid
    python3 validate_layout.py scene_state.json --fix     # also modify in-place

Checks:
    1. Every furniture AABB lies entirely inside the room bounds.
    2. No two furniture AABBs overlap.

With --fix:
    - Out-of-bounds furniture is clamped: pos = clamp(pos, dim/2, room - dim/2)
    - Colliding pairs: the SECOND furniture is pushed along the axis of
      smallest required separation. Iterates up to 20 passes.
    - If a furniture is larger than the room or collisions can't be
      resolved within 20 iterations, the script reports failure and
      asks the agent to shrink dimensions instead.

Exit codes:
    0 = scene is valid (after --fix if applied)
    1 = scene is invalid (and --fix could not fully resolve)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────
# Scene parsing — tolerates multiple schema variants
# ─────────────────────────────────────────────────────────────────────────

def get_room_bounds(scene: dict) -> tuple[float, float]:
    """Return (width, depth) of the scene's outer bounds. Handles:
        - scene.scene.bounds (canonical, SKILL.md schema invariant)
        - scene.bounds
        - rooms[0].bounds / dimensions (fallback)
    Returns (0.0, 0.0) if unrecoverable.
    """
    def _wd(b):
        if isinstance(b, dict):
            w = b.get("width") or b.get("w") or b.get("size_x")
            d = b.get("depth") or b.get("length") or b.get("d") or b.get("size_y")
            try:
                return float(w), float(d)
            except (TypeError, ValueError):
                return 0.0, 0.0
        if isinstance(b, list) and len(b) >= 2:
            try:
                return float(b[0]), float(b[1])
            except (TypeError, ValueError):
                return 0.0, 0.0
        return 0.0, 0.0

    sc = scene.get("scene")
    if isinstance(sc, dict):
        for k in ("bounds", "dimensions"):
            w, d = _wd(sc.get(k))
            if w > 0 and d > 0:
                return w, d
    for k in ("bounds", "dimensions"):
        w, d = _wd(scene.get(k))
        if w > 0 and d > 0:
            return w, d
    # fallback: largest containing rectangle across rooms
    rooms = scene.get("rooms") or []
    if isinstance(rooms, list):
        max_x = max_y = 0.0
        for r in rooms:
            if not isinstance(r, dict):
                continue
            for k in ("bounds", "dimensions", "dims_m"):
                rw, rd = _wd(r.get(k))
                if rw > 0 and rd > 0:
                    rb = r.get("bounds") if isinstance(r.get("bounds"), dict) else {}
                    rx = float(rb.get("x", 0) or 0)
                    ry = float(rb.get("y", 0) or 0)
                    max_x = max(max_x, rx + rw)
                    max_y = max(max_y, ry + rd)
                    break
        if max_x > 0 and max_y > 0:
            return max_x, max_y
    return 0.0, 0.0


def iter_furniture_refs(scene: dict) -> list[dict]:
    """Return a flat list of mutable furniture dicts (scene-level + nested-in-rooms)."""
    out: list[dict] = []
    if isinstance(scene.get("furniture"), list):
        for f in scene["furniture"]:
            if isinstance(f, dict):
                out.append(f)
    if isinstance(scene.get("rooms"), list):
        for r in scene["rooms"]:
            if isinstance(r, dict) and isinstance(r.get("furniture"), list):
                for f in r["furniture"]:
                    if isinstance(f, dict):
                        out.append(f)
    return out


def aabb_of(f: dict) -> tuple[float, float, float, float] | None:
    """Return (cx, cy, w, d) from a furniture dict, or None if unparseable."""
    pos = f.get("position")
    dim = f.get("dimensions") or f.get("size")
    if not (isinstance(pos, list) and len(pos) >= 2 and
            isinstance(dim, list) and len(dim) >= 2):
        return None
    try:
        return (float(pos[0]), float(pos[1]), float(dim[0]), float(dim[1]))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────
# Constraint checks
# ─────────────────────────────────────────────────────────────────────────

def bound_violations(cx, cy, w, d, W, D, eps=1e-6):
    """Return list of (side, magnitude_m) for any out-of-bounds excursion."""
    out = []
    if cx - w/2 < -eps:
        out.append(("LEFT",  -(cx - w/2)))
    if cy - d/2 < -eps:
        out.append(("FRONT", -(cy - d/2)))
    if cx + w/2 > W + eps:
        out.append(("RIGHT", (cx + w/2) - W))
    if cy + d/2 > D + eps:
        out.append(("BACK",  (cy + d/2) - D))
    return out


def clamp_to_bounds(cx, cy, w, d, W, D):
    """Snap (cx, cy) into the room. Returns (new_cx, new_cy) or (None, None) if too big."""
    if w > W + 1e-6 or d > D + 1e-6:
        return None, None
    return (max(w/2, min(cx, W - w/2)),
            max(d/2, min(cy, D - d/2)))


def aabb_overlap_area(a, b):
    """a, b = (cx, cy, w, d). Returns overlap area (0 if disjoint)."""
    ax0, ax1 = a[0] - a[2]/2, a[0] + a[2]/2
    ay0, ay1 = a[1] - a[3]/2, a[1] + a[3]/2
    bx0, bx1 = b[0] - b[2]/2, b[0] + b[2]/2
    by0, by1 = b[1] - b[3]/2, b[1] + b[3]/2
    return max(0, min(ax1, bx1) - max(ax0, bx0)) * \
           max(0, min(ay1, by1) - max(ay0, by0))


def push_apart_b(a, b, W, D):
    """Push b away from a along the smaller-effort axis. Clamped to room."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    min_dx = (a[2] + b[2]) / 2
    min_dy = (a[3] + b[3]) / 2
    # Required push along each axis (positive = magnitude)
    push_x = min_dx - abs(dx) if abs(dx) < min_dx else 0
    push_y = min_dy - abs(dy) if abs(dy) < min_dy else 0
    if push_x == 0 and push_y == 0:
        return b[0], b[1]
    # Pick axis with smaller push (less disruption)
    if push_x > 0 and (push_y == 0 or push_x <= push_y):
        sign = 1 if dx >= 0 else -1
        new_cx = a[0] + sign * min_dx
        new_cy = b[1]
    else:
        sign = 1 if dy >= 0 else -1
        new_cx = b[0]
        new_cy = a[1] + sign * min_dy
    # Clamp to room
    new_cx = max(b[2]/2, min(new_cx, W - b[2]/2))
    new_cy = max(b[3]/2, min(new_cy, D - b[3]/2))
    return new_cx, new_cy


# ─────────────────────────────────────────────────────────────────────────
# Top-level
# ─────────────────────────────────────────────────────────────────────────

def validate(scene: dict) -> tuple[list, list, float, float]:
    """Return (bound_issues, collisions, W, D).

    bound_issues: list of (idx, type, aabb, [(side, magnitude), ...])
    collisions:   list of (i, j, overlap_area_m2)
    """
    W, D = get_room_bounds(scene)
    if W <= 0 or D <= 0:
        return [], [], W, D
    items = iter_furniture_refs(scene)
    aabbs = [aabb_of(f) for f in items]

    bound_issues = []
    for i, (f, ab) in enumerate(zip(items, aabbs)):
        if ab is None:
            continue
        bv = bound_violations(*ab, W, D)
        if bv:
            bound_issues.append((i, f.get("type", "?"), ab, bv))

    collisions = []
    for i in range(len(items)):
        if aabbs[i] is None:
            continue
        for j in range(i + 1, len(items)):
            if aabbs[j] is None:
                continue
            ov = aabb_overlap_area(aabbs[i], aabbs[j])
            if ov > 1e-6:
                collisions.append((i, j, ov))
    return bound_issues, collisions, W, D


def fix_scene_in_place(scene: dict, max_iter: int = 20) -> tuple[bool, list[str]]:
    """Mutate scene to satisfy bounds + collision constraints.

    Returns (fully_fixed, log_lines).
    """
    log = []
    W, D = get_room_bounds(scene)
    if W <= 0 or D <= 0:
        return False, ["invalid room bounds — cannot fix"]
    items = iter_furniture_refs(scene)

    # 1. Bound clamping
    for i, f in enumerate(items):
        ab = aabb_of(f)
        if ab is None:
            continue
        cx, cy, w, d = ab
        new_cx, new_cy = clamp_to_bounds(cx, cy, w, d, W, D)
        if new_cx is None:
            log.append(f"  [{i}] {f.get('type','?')} dim ({w:.2f}×{d:.2f}) "
                       f"larger than room ({W:.2f}×{D:.2f}) — SHRINK dimensions")
            continue
        if abs(new_cx - cx) > 1e-3 or abs(new_cy - cy) > 1e-3:
            f["position"][0] = round(new_cx, 3)
            f["position"][1] = round(new_cy, 3)
            log.append(f"  [{i}] {f.get('type','?')} clamped "
                       f"({cx:.2f},{cy:.2f}) → ({new_cx:.2f},{new_cy:.2f})")

    # 2. Greedy collision resolution
    for it in range(max_iter):
        aabbs = [aabb_of(f) for f in items]
        cols = []
        for i in range(len(items)):
            if aabbs[i] is None:
                continue
            for j in range(i + 1, len(items)):
                if aabbs[j] is None:
                    continue
                if aabb_overlap_area(aabbs[i], aabbs[j]) > 1e-6:
                    cols.append((i, j))
        if not cols:
            break
        for i, j in cols:
            new_cx, new_cy = push_apart_b(aabbs[i], aabbs[j], W, D)
            items[j]["position"][0] = round(new_cx, 3)
            items[j]["position"][1] = round(new_cy, 3)
            log.append(f"  iter{it}: push [{j}] {items[j].get('type','?')} → ({new_cx:.2f},{new_cy:.2f})")

    # Final check
    bound2, cols2, *_ = validate(scene)
    return (not bound2 and not cols2), log


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene_path", help="Path to scene_state.json")
    ap.add_argument("--fix", action="store_true",
                    help="Modify the JSON in place (clamp + push-apart)")
    args = ap.parse_args()

    path = Path(args.scene_path)
    scene = json.loads(path.read_text())

    bound_issues, collisions, W, D = validate(scene)
    n_furn = len(iter_furniture_refs(scene))

    if W <= 0 or D <= 0:
        print(f"ERROR: cannot determine room bounds (got W={W} D={D})")
        sys.exit(1)

    if not bound_issues and not collisions:
        print(f"OK: room {W:.1f}×{D:.1f} m, {n_furn} furniture; all in-bounds and collision-free")
        sys.exit(0)

    print(f"INVALID: room {W:.1f}×{D:.1f} m, {n_furn} furniture")
    if bound_issues:
        print(f"  out-of-bounds ({len(bound_issues)}):")
        for i, ft, ab, bv in bound_issues:
            sides = ", ".join(f"{s} by {a:.2f} m" for s, a in bv)
            print(f"    [{i}] {ft} @ ({ab[0]:.1f},{ab[1]:.1f}) "
                  f"size ({ab[2]:.1f}×{ab[3]:.1f}): {sides}")
    if collisions:
        print(f"  collisions ({len(collisions)}):")
        for i, j, ov in collisions:
            print(f"    [{i}] ↔ [{j}]: {ov:.3f} m² overlap")

    if not args.fix:
        print("\nRun with --fix to clamp + resolve automatically.")
        sys.exit(1)

    print("\n--- Applying --fix ---")
    ok, log = fix_scene_in_place(scene)
    for ln in log:
        print(ln)
    path.write_text(json.dumps(scene, indent=2))
    print(f"\nSaved → {path}")
    if ok:
        print("✓ Fully fixed; scene is now in-bounds and collision-free.")
        sys.exit(0)
    print("⚠ Could NOT fully fix — likely furniture too large for room; "
          "shrink dimensions and re-run.")
    sys.exit(1)


if __name__ == "__main__":
    main()
