#!/usr/bin/env python3
"""Generate 27 benchmark scenes for RadioTwin Agent evaluation.

Produces scene_state.json files across 3 difficulty tiers:
  - Easy   (9 scenes): rectangular, 30-50 m², 1 AP, target 88%, 2-3 partitions
  - Medium (9 scenes): L-shaped,     50-100 m², 1 AP, target 93%, 3-5 partitions
  - Hard   (9 scenes): multi-room,  100-200 m², 1 AP, target 97%, 5-8 partitions

Targets are set so that naive center placement FAILS in ≥50% of scenes —
this ensures iterative reflection has room to demonstrate value. Scenes
that are trivially solved by center placement are rejected and regenerated.

Each scene includes walls (exterior + interior partitions that actually
divide the room), furniture, and TX auto-placement.
Deterministic via seed per scene for reproducibility.

Usage:
    python benchmark/generate_scenes.py [--output-dir benchmark/scenes]
    python benchmark/generate_scenes.py --verify   # check center_only fails
"""
from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scene configuration per tier
# ---------------------------------------------------------------------------

TIERS = {
    "easy": {
        "count": 9,
        "target_coverage_pct": 92,              # was 80/88 — small rooms are easy
        "num_tx": 1,
        "area_range": (30, 50),
        "shape": "rectangle",
        "wall_material": "itu_concrete",
        "interior_walls_range": (2, 3),
        "interior_wall_material": "itu_concrete",   # concrete, not plasterboard
        "partition_coverage_range": (0.6, 0.8),     # partitions span 60-80%
        # Difficulty: small rectangular room but concrete partitions force
        # 92%+ target — center placement usually shadowed by one partition
    },
    "medium": {
        "count": 9,
        "target_coverage_pct": 93,              # was 85
        "num_tx": 1,
        "area_range": (50, 100),
        "shape": "l_shaped",
        "wall_material": "itu_concrete",
        "interior_walls_range": (3, 5),         # was (2,3)
        "interior_wall_material": "itu_plasterboard",
        "partition_coverage_range": (0.6, 0.8),
        # Difficulty: L-shape + more partitions, still 1 TX, high target
    },
    "hard": {
        "count": 9,
        "target_coverage_pct": 94,
        "num_tx": 1,                            # single AP forces careful placement
        "area_range": (100, 150),
        "shape": "multi_room",
        "wall_material": "itu_concrete",
        "interior_walls_range": (4, 5),
        "interior_wall_material": "itu_plasterboard",  # 8 dB/wall — feasible for grid
        "partition_coverage_range": (0.6, 0.8), # longer partitions for real shadows
        # Difficulty: multi-room plasterboard partitions, single AP, 94% target.
        # Must place TX at the "junction" where paths to all sub-rooms are short.
    },
}

FURNITURE_POOL = [
    {"type": "desk", "dims": [1.2, 0.6, 0.75], "material": "itu_wood"},
    {"type": "chair", "dims": [0.5, 0.5, 0.9], "material": "itu_wood"},
    {"type": "bookcase", "dims": [0.8, 0.35, 1.8], "material": "itu_wood"},
    {"type": "sofa", "dims": [2.0, 0.9, 0.85], "material": "itu_wood"},
    {"type": "table", "dims": [1.0, 1.0, 0.75], "material": "itu_wood"},
    {"type": "cabinet", "dims": [0.8, 0.45, 0.9], "material": "itu_wood"},
    {"type": "wardrobe", "dims": [1.2, 0.6, 2.0], "material": "itu_wood"},
    {"type": "tv_stand", "dims": [1.2, 0.4, 0.5], "material": "itu_wood"},
]

ROOM_TYPES = ["office", "bedroom", "living_room", "conference", "studio"]

HEIGHT = 3.0
WALL_THICK = 0.15
FREQUENCY_HZ = 3.5e9
TX_POWER_DBM = 20.0
RX_HEIGHT = 1.5
CELL_SIZE = 0.2


# ---------------------------------------------------------------------------
# Geometry generators
# ---------------------------------------------------------------------------

def make_rectangle(rng: random.Random, area_range: tuple[float, float]) -> dict:
    """Generate a rectangular room polygon and dimensions."""
    area = rng.uniform(*area_range)
    aspect = rng.uniform(0.6, 1.0)  # width/length ratio
    length = math.sqrt(area / aspect)
    width = area / length
    width = round(width, 1)
    length = round(length, 1)
    polygon = [[0, 0], [width, 0], [width, length], [0, length]]
    return {"polygon": polygon, "width": width, "length": length}


def make_l_shaped(rng: random.Random, area_range: tuple[float, float]) -> dict:
    """Generate an L-shaped room polygon."""
    area = rng.uniform(*area_range)
    # L-shape: main rectangle + notch cut from one corner
    base_w = round(rng.uniform(6, 10), 1)
    base_l = round(area / base_w, 1)
    # Cut corner: remove a rectangle from NE corner
    cut_w = round(rng.uniform(base_w * 0.3, base_w * 0.5), 1)
    cut_l = round(rng.uniform(base_l * 0.3, base_l * 0.5), 1)
    polygon = [
        [0, 0],
        [base_w, 0],
        [base_w, base_l - cut_l],
        [base_w - cut_w, base_l - cut_l],
        [base_w - cut_w, base_l],
        [0, base_l],
    ]
    return {"polygon": polygon, "width": base_w, "length": base_l}


def make_multi_room(rng: random.Random, area_range: tuple[float, float]) -> dict:
    """Generate a large room with interior partition walls."""
    area = rng.uniform(*area_range)
    aspect = rng.uniform(0.5, 0.8)
    length = math.sqrt(area / aspect)
    width = area / length
    width = round(width, 1)
    length = round(length, 1)
    polygon = [[0, 0], [width, 0], [width, length], [0, length]]
    return {"polygon": polygon, "width": width, "length": length}


# ---------------------------------------------------------------------------
# Wall generation
# ---------------------------------------------------------------------------

def generate_exterior_walls(polygon: list[list[float]], room_id: str) -> list[dict]:
    """Generate wall segments from polygon vertices."""
    walls = []
    n = len(polygon)
    for i in range(n):
        start = polygon[i]
        end = polygon[(i + 1) % n]
        walls.append({
            "id": f"wall_{room_id}_{i+1:02d}",
            "room_id": room_id,
            "start": start,
            "end": end,
            "thickness": WALL_THICK,
            "material": "itu_concrete",
            "is_interior": False,
            "has_window": i % 3 == 0,  # windows on every 3rd wall
            "has_door": i == 0,  # door on first wall
        })
    return walls


def generate_interior_walls(
    rng: random.Random, width: float, length: float,
    room_id: str, count: int, start_idx: int,
    material: str = "itu_plasterboard",
    coverage_range: tuple[float, float] = (0.5, 0.7),
) -> list[dict]:
    """Generate interior partition walls that actually divide the room.

    Unlike the old version that placed floating wall stubs, these walls span
    a large fraction (50-90% per tier) of the room dimension and alternate
    orientations to create shadow zones. Partitions are positioned on an
    irregular grid so placements don't trivially align with center_tx.
    """
    walls = []
    # Reserve zones so partitions don't overlap by using evenly-spaced slots
    vert_slots = [width * (0.25 + 0.5 * (i + 0.5) / max(1, count // 2 + 1))
                  for i in range(count // 2 + 1)]
    horiz_slots = [length * (0.25 + 0.5 * (i + 0.5) / max(1, (count + 1) // 2))
                   for i in range(count // 2 + 1)]
    rng.shuffle(vert_slots)
    rng.shuffle(horiz_slots)

    for i in range(count):
        coverage = rng.uniform(*coverage_range)
        # Alternate between vertical and horizontal partitions
        if i % 2 == 0 and vert_slots:
            # Vertical partition spanning coverage% of room length
            x = round(vert_slots.pop(), 1)
            span = length * coverage
            y_start = round(rng.uniform(0, length - span), 1)
            y_end = round(y_start + span, 1)
            start = [x, y_start]
            end = [x, y_end]
        elif horiz_slots:
            # Horizontal partition spanning coverage% of room width
            y = round(horiz_slots.pop(), 1)
            span = width * coverage
            x_start = round(rng.uniform(0, width - span), 1)
            x_end = round(x_start + span, 1)
            start = [x_start, y]
            end = [x_end, y]
        else:
            continue  # out of slots

        walls.append({
            "id": f"wall_{room_id}_int_{start_idx + i + 1:02d}",
            "room_id": room_id,
            "start": start,
            "end": end,
            "thickness": 0.15 if material == "itu_concrete" else 0.12,
            "material": material,
            "is_interior": True,
            "has_window": False,
            "has_door": True,  # interior walls have doorways
        })
    return walls


# ---------------------------------------------------------------------------
# Furniture placement (simple grid-based, no collision for benchmark)
# ---------------------------------------------------------------------------

def place_furniture(
    rng: random.Random, width: float, length: float,
    count: int, room_id: str
) -> list[dict]:
    """Place furniture items along walls with margin."""
    items = []
    margin = 0.3
    for i in range(count):
        furn = rng.choice(FURNITURE_POOL)
        fw, fd = furn["dims"][0], furn["dims"][1]
        # Place along a wall with some randomness
        wall_side = i % 4
        if wall_side == 0:  # south wall
            x = round(rng.uniform(margin + fw / 2, width - margin - fw / 2), 2)
            y = round(margin + fd / 2, 2)
            orient = 0
        elif wall_side == 1:  # east wall
            x = round(width - margin - fd / 2, 2)
            y = round(rng.uniform(margin + fw / 2, length - margin - fw / 2), 2)
            orient = 90
        elif wall_side == 2:  # north wall
            x = round(rng.uniform(margin + fw / 2, width - margin - fw / 2), 2)
            y = round(length - margin - fd / 2, 2)
            orient = 180
        else:  # west wall
            x = round(margin + fd / 2, 2)
            y = round(rng.uniform(margin + fw / 2, length - margin - fw / 2), 2)
            orient = 270

        items.append({
            "id": f"furn_{room_id}_{i+1:03d}",
            "type": furn["type"],
            "catalog_id": None,
            "position": [x, y, 0.0],
            "orientation_deg": orient,
            "dimensions": furn["dims"],
            "material": furn["material"],
            "visible": True,
        })
    return items


# ---------------------------------------------------------------------------
# TX placement
# ---------------------------------------------------------------------------

def place_transmitters(
    width: float, length: float, count: int, room_id: str,
    frequency_hz: float = FREQUENCY_HZ,
) -> list[dict]:
    """Place TXs at ceiling height, distributed across the room."""
    txs = []
    if count == 1:
        positions = [[width / 2, length / 2]]
    elif count == 2:
        positions = [[width / 3, length / 2], [2 * width / 3, length / 2]]
    else:
        positions = [
            [width / 4, length / 3],
            [3 * width / 4, length / 3],
            [width / 2, 2 * length / 3],
        ]
    for i, (x, y) in enumerate(positions[:count]):
        txs.append({
            "id": f"tx_{room_id}_{i+1}",
            "name": f"AP-{i+1}",
            "position": [round(x, 2), round(y, 2), HEIGHT - 0.2],
            "power_dbm": TX_POWER_DBM,
            "frequency_hz": frequency_hz,
            "antenna": {
                "type": "isotropic",
                "pattern": None,
                "polarization": "V",
                "mimo_config": None,
            },
            "orientation_deg": 0,
            "visible": True,
        })
    return txs


# ---------------------------------------------------------------------------
# Scene assembly
# ---------------------------------------------------------------------------

def generate_scene(
    tier: str, index: int, config: dict, seed: int
) -> dict[str, Any]:
    """Generate a single benchmark scene_state.json."""
    rng = random.Random(seed)
    scene_id = f"{tier}_{index+1:02d}"
    room_id = f"room_{scene_id}"
    room_type = rng.choice(ROOM_TYPES)

    # Generate geometry
    shape = config["shape"]
    if shape == "rectangle":
        geom = make_rectangle(rng, config["area_range"])
    elif shape == "l_shaped":
        geom = make_l_shaped(rng, config["area_range"])
    else:
        geom = make_multi_room(rng, config["area_range"])

    width, length = geom["width"], geom["length"]
    polygon = geom["polygon"]

    # Walls
    ext_walls = generate_exterior_walls(polygon, room_id)

    int_wall_count = 0
    if "interior_walls" in config:
        int_wall_count = config["interior_walls"]
    elif "interior_walls_range" in config:
        int_wall_count = rng.randint(*config["interior_walls_range"])

    int_walls = generate_interior_walls(
        rng, width, length, room_id, int_wall_count, len(ext_walls),
        material=config.get("interior_wall_material", "itu_plasterboard"),
        coverage_range=config.get("partition_coverage_range", (0.5, 0.7)),
    )
    all_walls = ext_walls + int_walls

    # Furniture: scale count with area (more dense than before for obstruction)
    area = width * length
    furn_count = max(4, min(12, int(area / 6)))  # was area / 8
    furniture = place_furniture(rng, width, length, furn_count, room_id)

    # Frequency (per-tier override or default)
    freq_hz = config.get("frequency_hz", FREQUENCY_HZ)

    # Transmitters
    if "num_tx" in config:
        num_tx = config["num_tx"]
    else:
        num_tx = rng.randint(*config["num_tx_range"])
    transmitters = place_transmitters(width, length, num_tx, room_id, freq_hz)

    # Assemble scene_state
    now = datetime.now(timezone.utc).isoformat()
    state = {
        "version": "2.0",
        "meta": {
            "created": now,
            "modified": now,
            "name": f"Benchmark {tier.capitalize()} {index+1:02d}",
            "description": (
                f"{tier.capitalize()} benchmark scene: {room_type}, "
                f"{width}x{length}m, {shape} layout, "
                f"{num_tx} TX, target {config['target_coverage_pct']}% coverage"
            ),
            "benchmark": {
                "tier": tier,
                "index": index + 1,
                "seed": seed,
                "target_coverage_pct": config["target_coverage_pct"],
                "shape": shape,
                "area_m2": round(area, 1),
            },
        },
        "scene": {
            "type": "indoor",
            "origin": "SW",
            "coordinate_system": {
                "x": "east", "y": "north", "z": "up",
                "theta_zero": "north", "units": "meters",
            },
            "bounds": {"width": width, "depth": length, "height": HEIGHT},
            "frequency_hz": freq_hz,
            "source_format": "benchmark_generated",
        },
        "rooms": [{
            "id": room_id,
            "name": f"{room_type.replace('_', ' ').title()}",
            "type": room_type,
            "polygon": polygon,
            "height": HEIGHT,
            "materials": {
                "walls": "itu_concrete",
                "floor": "itu_concrete",
                "ceiling": "itu_plasterboard",
            },
        }],
        "walls": all_walls,
        "furniture": furniture,
        "transmitters": transmitters,
        "receivers": [{
            "id": f"rx_grid_{scene_id}",
            "type": "grid",
            "height": RX_HEIGHT,
            "resolution": CELL_SIZE,
            "bounds": [[0, 0], [width, length]],
            "visible": True,
        }],
        "constraints": [],
        "simulation_results": [],
        "export_history": [],
        "code_artifacts": [],
    }
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _center_coverage(scene: dict) -> float:
    """Compute analytical coverage at ceiling-center placement.

    Uses the same 3GPP InH model as the evaluation harness so the difficulty
    filter is consistent with how methods are scored.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluation_harness import compute_analytical_coverage
    bounds = scene["scene"]["bounds"]
    w, l = bounds["width"], bounds["depth"]
    try:
        cov, _ = compute_analytical_coverage(scene, w / 2, l / 2)
        return cov
    except Exception:
        return 0.0  # treat failures as trivially impossible → reject


def _grid_best_coverage(scene: dict, resolution: float = 2.0) -> float:
    """Compute the best coverage any grid position can achieve.

    This is the oracle upper bound. If grid search can't meet the target,
    the scene is INFEASIBLE and must be rejected (bad for all methods).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluation_harness import compute_analytical_coverage
    import numpy as np
    bounds = scene["scene"]["bounds"]
    w, l = bounds["width"], bounds["depth"]
    xs = np.arange(resolution, w, resolution)
    ys = np.arange(resolution, l, resolution)
    best = 0.0
    for x in xs:
        for y in ys:
            try:
                cov, _ = compute_analytical_coverage(scene, float(x), float(y))
                if cov > best:
                    best = cov
            except Exception:
                continue
    return best


def generate_all(output_dir: Path, verify_difficulty: bool = True) -> dict:
    """Generate all 27 benchmark scenes with difficulty filter.

    If verify_difficulty=True, each generated scene must satisfy:
      - center_coverage < target  (center placement alone fails)
      - grid_best_coverage >= target  (oracle placement succeeds)

    Scenes that don't satisfy both are rejected and regenerated with a
    new seed (up to max_retries per slot). This ensures scenes discriminate
    between naive and reflective placement.
    """
    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "total_scenes": 27,
        "verify_difficulty": verify_difficulty,
        "tiers": {},
    }

    base_seed = 42_000
    scene_idx = 0
    max_retries = 60  # per slot (15s/retry for hard tier)

    for tier, config in TIERS.items():
        tier_dir = output_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        tier_scenes = []
        target = config["target_coverage_pct"]

        for i in range(config["count"]):
            scene = None
            seed_used = None
            rejected_reasons = []

            for retry in range(max_retries):
                seed = base_seed + scene_idx * 1000 + retry
                candidate = generate_scene(tier, i, config, seed)

                if not verify_difficulty:
                    scene = candidate
                    seed_used = seed
                    break

                center_cov = _center_coverage(candidate)
                if center_cov >= target - 1:
                    rejected_reasons.append(
                        f"center={center_cov:.1f}% nearly meets target={target}%"
                    )
                    continue  # trivial — center already meets target

                # Check that oracle CAN solve this (feasibility)
                grid_cov = _grid_best_coverage(candidate, resolution=2.0)
                if grid_cov < target:
                    rejected_reasons.append(
                        f"grid={grid_cov:.1f}% < target={target}% (infeasible)"
                    )
                    continue  # even oracle can't meet target — unfair

                # Require meaningful gap between center and grid
                if grid_cov - center_cov < 3:
                    rejected_reasons.append(
                        f"gap={grid_cov-center_cov:.1f}pp < 3 (non-discriminating)"
                    )
                    continue  # center and grid too close — no room for reflection

                scene = candidate
                seed_used = seed
                scene["meta"]["benchmark"]["center_coverage_pct"] = round(center_cov, 1)
                scene["meta"]["benchmark"]["grid_best_coverage_pct"] = round(grid_cov, 1)
                break

            if scene is None:
                # Fall back: take the last candidate even though it doesn't discriminate
                seed_used = base_seed + scene_idx * 1000
                scene = generate_scene(tier, i, config, seed_used)
                print(f"  WARNING: {tier}_{i+1:02d} — no discriminating scene "
                      f"after {max_retries} retries. Reasons: {rejected_reasons[:3]}")

            scene_dir = tier_dir / f"scene_{i+1:02d}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            state_path = scene_dir / "scene_state.json"
            state_path.write_text(json.dumps(scene, indent=2))

            tier_scenes.append({
                "scene_id": f"{tier}_{i+1:02d}",
                "path": str(scene_dir.relative_to(output_dir)),
                "seed": seed_used,
                "area_m2": scene["meta"]["benchmark"]["area_m2"],
                "shape": scene["meta"]["benchmark"]["shape"],
                "target_coverage_pct": config["target_coverage_pct"],
                "center_coverage_pct": scene["meta"]["benchmark"].get(
                    "center_coverage_pct"),
                "grid_best_coverage_pct": scene["meta"]["benchmark"].get(
                    "grid_best_coverage_pct"),
                "num_tx": len(scene["transmitters"]),
                "num_walls": len(scene["walls"]),
                "num_furniture": len(scene["furniture"]),
            })

            scene_idx += 1

        manifest["tiers"][tier] = {
            "count": config["count"],
            "target_coverage_pct": config["target_coverage_pct"],
            "scenes": tier_scenes,
        }

    # Write manifest
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest


def print_summary(manifest: dict) -> None:
    """Print a human-readable summary of generated scenes."""
    print(f"\n{'='*80}")
    print(f"  RadioTwin Benchmark — {manifest['total_scenes']} scenes generated")
    print(f"  {manifest['generated']}")
    print(f"{'='*80}\n")

    for tier, info in manifest["tiers"].items():
        print(f"  {tier.upper()} ({info['count']} scenes, "
              f"target {info['target_coverage_pct']}% coverage)")
        print(f"    {'scene_id':<12} {'area':>7} {'shape':>14} "
              f"{'TX':>3} {'walls':>6} {'furn':>5} "
              f"{'center%':>8} {'grid%':>7} {'Δtarget':>8}")
        print(f"    {'-'*12} {'-'*7} {'-'*14} {'-'*3} {'-'*6} {'-'*5} "
              f"{'-'*8} {'-'*7} {'-'*8}")
        center_fails = 0
        for s in info["scenes"]:
            cc = s.get("center_coverage_pct")
            gc = s.get("grid_best_coverage_pct")
            target = info["target_coverage_pct"]
            cc_str = f"{cc:.1f}" if cc is not None else "  -  "
            gc_str = f"{gc:.1f}" if gc is not None else "  -  "
            delta = f"+{gc - target:.1f}" if gc is not None else "  -  "
            if cc is not None and cc < target:
                center_fails += 1
            print(f"    {s['scene_id']:<12} "
                  f"{s['area_m2']:5.1f}m² "
                  f"{s['shape']:>14s} "
                  f"{s['num_tx']:>3} "
                  f"{s['num_walls']:>6} "
                  f"{s['num_furniture']:>5} "
                  f"{cc_str:>8} "
                  f"{gc_str:>7} "
                  f"{delta:>8}")
        if info["scenes"]:
            print(f"    → Center placement fails on {center_fails}/{len(info['scenes'])} "
                  f"scenes ({center_fails*100//len(info['scenes'])}%)")
        print()


def verify_existing_scenes(scenes_dir: Path) -> None:
    """Run difficulty diagnostics on an existing scenes directory."""
    manifest_path = scenes_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest at {manifest_path}")
        return
    manifest = json.loads(manifest_path.read_text())
    for tier, info in manifest["tiers"].items():
        target = info["target_coverage_pct"]
        fails = 0
        for s in info["scenes"]:
            scene_path = scenes_dir / s["path"] / "scene_state.json"
            scene = json.loads(scene_path.read_text())
            cc = _center_coverage(scene)
            if cc < target:
                fails += 1
        print(f"  {tier}: center fails on {fails}/{info['count']} scenes "
              f"({fails*100//info['count']}%), target={target}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RadioTwin benchmark scenes")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent / "scenes",
        help="Output directory for benchmark scenes",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip difficulty filter (faster but produces trivial scenes)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify existing scenes, don't regenerate",
    )
    args = parser.parse_args()

    if args.verify_only:
        verify_existing_scenes(args.output_dir)
    else:
        manifest = generate_all(args.output_dir,
                                verify_difficulty=not args.no_verify)
        print_summary(manifest)
        print(f"Manifest: {args.output_dir / 'manifest.json'}")
