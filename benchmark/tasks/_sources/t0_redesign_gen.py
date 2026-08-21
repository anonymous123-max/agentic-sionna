"""Generator for T0 scene_gen redesigned 100-task suite.

Output: benchmark/tasks/_sources/t0_redesign.json
Schema: matches benchmark/build_tasks.py unified schema; tier=T0_scene_gen;
ids T0E001..T0E060 (easy) and T0H001..T0H040 (hard).

Verifier strategy: every task is a `composite` with grep-only subchecks
(file_exists + collision_free_check + in_bounds_check are real handlers;
all others fall through to `_check_generic_tokens` keyword search per
verifier.py:332).
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "t0_redesign.json"


def task(id, capability, difficulty, split, name, prompt, distractor,
         assertions, grep_tokens):
    """Build one T0 task with composite verifier."""
    return {
        "id": id,
        "origin": "redesign",
        "origin_id": id,
        "tier": "T0_scene_gen",
        "capability": capability,
        "difficulty": difficulty,
        "split": split,
        "name": name,
        "prompt": prompt,
        "distractor": distractor,
        "scene_path": None,
        "required_artifacts": ["scene_state.json"],
        "assertions": assertions,
        "verifier": {
            "type": "composite",
            "subchecks": [
                {"key": "must_create_scene_state", "type": "file_exists"},
                {"metric": "collision_free_check", "type": "code_contains"},
                {"metric": "in_bounds_check", "type": "code_contains"},
            ] + [{"metric": tok, "type": "code_contains"} for tok in grep_tokens],
        },
    }


def indoor_easy(suffix, name, prompt, distractor, dims, furniture,
                extra_tokens=None):
    """scene_indoor easy template: rectangular single room with furniture."""
    w, d = dims
    id = f"T0E{suffix:03d}"
    split = "train" if suffix <= 13 else "test"
    grep = ["_".join(f.replace(" ", "_").replace("-", "_") for f in furniture)]
    if extra_tokens:
        grep += extra_tokens
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect expected_furniture={furniture}",
        f"expect min_furniture={len(furniture)}",
        f"expect room_dims_range={{'width': [{w-0.5}, {w+0.5}], 'depth': [{d-0.5}, {d+0.5}]}}",
    ]
    return task(id, "scene_indoor", "easy", split, name, prompt, distractor,
                assertions, grep)


TASKS = []

# ════════════════════════════════════════════════════════════════════════
# scene_indoor easy (22)   T0E001–T0E013 train · T0E014–T0E022 test
# ════════════════════════════════════════════════════════════════════════

TASKS.append(indoor_easy(1, "Compact home office",
    "Create a 4 m × 3 m home office. Place one desk against a wall, one office chair facing the desk, and one bookshelf along another wall. Use default RF materials.",
    "Wrong: placing the chair clipping into the desk produces collision overlaps that fail RT path computation. Right: position the chair centroid 0.4–0.6 m in front of the desk's long edge, leaving sit-down clearance.",
    (4, 3), ["desk", "office chair", "bookshelf"]))

TASKS.append(indoor_easy(2, "Single bedroom",
    "Create a 4 m × 3.5 m single bedroom. Place one single bed (1 m × 2 m) against a long wall, one nightstand beside the bed, and one dresser along the opposite wall. Default RF materials.",
    "Wrong: putting the nightstand on the wall side of the bed where there is no clearance produces overlap with the bed; right: place the nightstand on the door-side of the bed with 0.05–0.1 m gap.",
    (4, 3.5), ["single bed", "nightstand", "dresser"]))

TASKS.append(indoor_easy(3, "Reading nook",
    "Create a 3 m × 3 m reading nook. Include one armchair, one floor lamp, and one side table arranged so the lamp is within 0.6 m of the armchair. Default materials.",
    "Wrong: placing the floor lamp at the room centroid leaves the chair too far from light; right: place the lamp behind or to the side of the armchair within 0.6 m so reading light reaches the chair.",
    (3, 3), ["armchair", "floor lamp", "side table"]))

TASKS.append(indoor_easy(4, "Small kitchenette",
    "Create a 4 m × 3 m kitchenette. Place an L-arrangement of one kitchen counter (2 m × 0.6 m), one refrigerator (0.7 m × 0.7 m), and one sink unit (0.6 m × 0.6 m) along two adjacent walls. Default materials.",
    "Wrong: the refrigerator placed in front of the counter blocks workflow and overlaps the counter footprint; right: align counter, sink, and fridge along two adjacent walls forming an L so each appliance sits flush against a wall.",
    (4, 3), ["kitchen counter", "refrigerator", "sink unit"]))

TASKS.append(indoor_easy(5, "Music practice room",
    "Create a 4 m × 3 m music practice room. Place one upright piano (1.5 m × 0.6 m) flush against a wall, one piano bench in front of it, and one music stand to the right of the bench. Default materials.",
    "Wrong: putting the piano in the middle of the room makes the bench overlap the piano keyboard area; right: place the piano against the long wall and the bench 0.4–0.5 m in front of the keyboard.",
    (4, 3), ["upright piano", "piano bench", "music stand"]))

TASKS.append(indoor_easy(6, "Childs bedroom",
    "Create a 4 m × 3.5 m childs bedroom. Place one bunk bed (1 m × 2 m) along a wall, one dresser, and one toy chest near the foot of the bunk bed. Default materials.",
    "Wrong: stacking the toy chest where it blocks the bunk-bed ladder breaks egress; right: leave 0.6 m clear in front of the ladder side of the bunk bed.",
    (4, 3.5), ["bunk bed", "dresser", "toy chest"]))

TASKS.append(indoor_easy(7, "Walk-in closet",
    "Create a 3 m × 3 m walk-in closet. Place one clothing rack (1.5 m × 0.5 m) along a long wall, one shoe shelf (1 m × 0.4 m) along an adjacent wall, and one ottoman in the centre. Default materials.",
    "Wrong: placing the ottoman in front of the clothing rack blocks access; right: leave 0.5–0.7 m clearance in front of the rack and put the ottoman in the open quadrant.",
    (3, 3), ["clothing rack", "shoe shelf", "ottoman"]))

TASKS.append(indoor_easy(8, "Small dining room",
    "Create a 4 m × 4 m dining room with one round dining table (1.2 m diameter) centred in the room and four matching chairs arranged at 90° around it. Default materials.",
    "Wrong: pushing the table against a wall removes seating clearance on one side; right: centre the table so 0.6 m clearance is available for chairs on all four sides.",
    (4, 4), ["dining table", "chair"], extra_tokens=["four"]))

TASKS.append(indoor_easy(9, "Guest room",
    "Create a 4 m × 4 m guest room. Place one queen bed (1.5 m × 2 m) against a wall, one wardrobe along the opposite wall, and one luggage rack between them by the foot of the bed. Default materials.",
    "Wrong: putting the luggage rack at the head of the bed blocks the nightstand area; right: place the luggage rack at the foot of the bed against the wall with 0.4 m clearance.",
    (4, 4), ["queen bed", "wardrobe", "luggage rack"]))

TASKS.append(indoor_easy(10, "Mudroom",
    "Create a 3 m × 2.5 m mudroom. Place one bench (1 m × 0.4 m), one coat rack on a perpendicular wall, and one shoe rack (0.8 m × 0.3 m) under the bench wall. Default materials.",
    "Wrong: a coat rack on the same wall as the bench prevents wall-hung coats from clearing the seated bench occupant; right: place the coat rack on a wall perpendicular to the bench.",
    (3, 2.5), ["bench", "coat rack", "shoe rack"]))

TASKS.append(indoor_easy(11, "Reception desk area",
    "Create a 4 m × 3.5 m reception area. Place one reception desk (1.6 m × 0.7 m) facing the entry, one receptionist chair behind it, and one waiting bench (1.5 m × 0.5 m) along the wall opposite the entry. Default materials.",
    "Wrong: putting the waiting bench behind the reception desk hides waiting guests; right: place the bench along the wall opposite the entry so guests are visible from the desk.",
    (4, 3.5), ["reception desk", "receptionist chair", "waiting bench"]))

TASKS.append(indoor_easy(12, "Hobby crafts room",
    "Create a 4 m × 3 m crafts room. Place one craft table (1.5 m × 0.7 m) in the centre, one supply cabinet (1 m × 0.4 m) along a wall, and one craft stool tucked under the long side of the craft table. Default materials.",
    "Wrong: pushing the supply cabinet flush against the craft table creates overlap and blocks one workspace edge; right: keep the cabinet against a wall with 0.4–0.6 m walkway clearance.",
    (4, 3), ["craft table", "supply cabinet", "craft stool"]))

TASKS.append(indoor_easy(13, "Library",
    "Create a 5 m × 4 m home library. Place three bookshelves (1 m × 0.3 m each) along one long wall, one reading chair facing them, and one side table within 0.6 m of the chair. Default materials.",
    "Wrong: orienting bookshelves perpendicular to the wall blocks browsing access; right: align all three bookshelves flush against the long wall so each one is readable from the centre of the room.",
    (5, 4), ["bookshelf", "reading chair", "side table"], extra_tokens=["three"]))

# ─── easy test split (T0E014–T0E022) ───
TASKS.append(indoor_easy(14, "Yoga studio",
    "Create a 4 m × 4 m yoga studio. Place one yoga mat rack (1 m × 0.4 m) along one wall, one full-length wall mirror on an adjacent wall, and one meditation bench against the wall opposite the mirror. Default materials.",
    "Wrong: placing the meditation bench in front of the mirror obscures the practitioner's reflection; right: keep the mirror wall clear and place the bench on the far wall.",
    (4, 4), ["yoga mat rack", "wall mirror", "meditation bench"]))

TASKS.append(indoor_easy(15, "Server closet",
    "Create a 3 m × 3 m server closet. Place one server rack (0.6 m × 0.8 m) at the back wall centred, one KVM cart (0.5 m × 0.5 m) along a side wall, and one step ladder in a corner. Default materials.",
    "Wrong: placing the KVM cart directly in front of the server rack blocks cable access; right: keep ≥0.7 m clearance in front of the rack and place the KVM cart against a perpendicular wall.",
    (3, 3), ["server rack", "KVM cart", "step ladder"]))

TASKS.append(indoor_easy(16, "Breakroom",
    "Create a 4 m × 4 m office breakroom. Place one round table (1.2 m diameter) centred, four chairs around the table, and one vending machine flush against a wall. Default materials.",
    "Wrong: placing the vending machine in the centre of the room takes the prime gathering spot; right: put the vending machine flush against a wall and centre the table for seating.",
    (4, 4), ["round table", "chair", "vending machine"], extra_tokens=["four"]))

TASKS.append(indoor_easy(17, "Examination room",
    "Create a 4 m × 3 m medical examination room. Place one exam table (1.8 m × 0.7 m) along the long wall, one doctor stool beside it, and one supply cart against the opposite wall. Default materials.",
    "Wrong: positioning the supply cart at the head of the exam table blocks doctor access from above; right: keep the head of the exam table clear and the supply cart on the opposite wall within reach.",
    (4, 3), ["exam table", "doctor stool", "supply cart"]))

TASKS.append(indoor_easy(18, "Photo studio",
    "Create a 4 m × 4 m small photo studio. Place one backdrop stand (2 m × 0.3 m) along the back wall, one light stand at 60° to the backdrop, and one tripod stool for the photographer. Default materials.",
    "Wrong: putting the light stand directly between the camera and the backdrop creates a hotspot; right: angle the light at 30–60° from the camera-subject axis.",
    (4, 4), ["backdrop stand", "light stand", "tripod stool"]))

TASKS.append(indoor_easy(19, "Drying room",
    "Create a 3 m × 3 m laundry/drying room. Place one drying rack (1 m × 0.5 m), one washing machine (0.6 m × 0.6 m), and one dryer (0.6 m × 0.6 m) side-by-side along one wall. Default materials.",
    "Wrong: placing the drying rack directly over the washing machine creates clearance issues for loading; right: place the rack on a perpendicular wall with ≥0.7 m clearance to washer/dryer doors.",
    (3, 3), ["drying rack", "washing machine", "dryer"]))

TASKS.append(indoor_easy(20, "Small lab",
    "Create a 5 m × 4 m lab. Place one workbench (2 m × 0.7 m) along a long wall, one lab stool in front of the bench, and one fume hood (1 m × 0.7 m) on the perpendicular wall. Default materials.",
    "Wrong: placing the fume hood opposite the workbench requires crossing the room with chemicals; right: place the fume hood on a wall adjacent to the workbench so the user pivots without moving.",
    (5, 4), ["workbench", "lab stool", "fume hood"]))

TASKS.append(indoor_easy(21, "Music listening room",
    "Create a 4 m × 4 m audiophile listening room. Place one armchair centred 2.5 m from the back wall, one hi-fi cabinet (1.2 m × 0.4 m) on the front wall, and two speaker stands flanking the cabinet 1.5 m apart. Default materials.",
    "Wrong: placing the armchair against the back wall causes nulls in low-frequency response; right: pull the chair 0.4–0.8 m off the back wall and form an equilateral triangle with the speakers.",
    (4, 4), ["armchair", "hi-fi cabinet", "speaker stand"], extra_tokens=["two"]))

TASKS.append(indoor_easy(22, "Boot room",
    "Create a 3 m × 2.5 m boot room. Place one boot bench (1 m × 0.4 m) against a long wall, one coat rack on an adjacent wall, and one umbrella stand in a corner. Default materials.",
    "Wrong: putting the umbrella stand in front of the boot bench blocks the seating area; right: place the umbrella stand in a corner where wet umbrellas drain onto the floor without splashing the bench.",
    (3, 2.5), ["boot bench", "coat rack", "umbrella stand"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_l_shape easy (6)   T0E023–T0E026 train · T0E027–T0E028 test
# ════════════════════════════════════════════════════════════════════════

def l_easy(suffix, name, prompt, distractor, long_dims, short_dims, furniture,
           extra_tokens=None):
    id = f"T0E{suffix:03d}"
    split = "train" if suffix <= 26 else "test"
    grep = ["l_shape", "_".join(f.replace(" ", "_").replace("-", "_") for f in furniture)]
    if extra_tokens:
        grep += extra_tokens
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect L-shape with long arm {long_dims[0]} m × {long_dims[1]} m and short arm {short_dims[0]} m × {short_dims[1]} m",
        f"expect expected_furniture={furniture}",
        f"expect min_furniture={len(furniture)}",
    ]
    return task(id, "scene_indoor_l_shape", "easy", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(l_easy(23, "Small L-shaped home office",
    "Create an L-shaped home office. Long arm 4 m × 3 m holds one desk against the outer wall, one office chair facing it; short arm 3 m × 2 m holds one filing cabinet against an outer wall. Default materials.",
    "Wrong: defining the L as two separate rooms with overlapping bounds at the corner produces a duplicated junction cell; right: define one polygon with six vertices that traces the L outline.",
    (4, 3), (3, 2), ["desk", "office chair", "filing cabinet"]))

TASKS.append(l_easy(24, "L-shaped studio (kitchen + living)",
    "Create an L-shaped studio. Long arm 5 m × 4 m is the living area with one sofa against the outer wall; short arm 3 m × 3 m is the kitchen with one kitchen counter and one stool at the counter. Default materials.",
    "Wrong: treating the inside corner as a wall blocks line of sight between zones; right: leave the inside corner open so the L is one continuous space.",
    (5, 4), (3, 3), ["sofa", "kitchen counter", "stool"]))

TASKS.append(l_easy(25, "L-shaped corridor with reading bench",
    "Create an L-shaped corridor. Long arm 5 m × 2 m, short arm 3 m × 2 m. Place one reading bench at the inner corner where the two arms meet, one coat rack along the long arm wall, and one umbrella stand at the short arm endpoint. Default materials.",
    "Wrong: placing the bench across the corridor narrows passage below code; right: place the bench flush against the inside corner so the 1 m walking lane is preserved.",
    (5, 2), (3, 2), ["reading bench", "coat rack", "umbrella stand"]))

TASKS.append(l_easy(26, "L-shaped bedroom + nook",
    "Create an L-shaped bedroom. Long arm 4 m × 3 m holds one double bed, one nightstand; short arm 2 m × 2 m is a reading nook with one armchair. Default materials.",
    "Wrong: placing the armchair facing the dead-end wall ignores the L's interior connection; right: orient the armchair to face the long arm so the occupant sees the rest of the room.",
    (4, 3), (2, 2), ["double bed", "nightstand", "armchair"]))

TASKS.append(l_easy(27, "L-shaped meeting + breakout",
    "Create an L-shaped meeting space. Long arm 5 m × 3 m holds one meeting table, six chairs; short arm 3 m × 3 m is a breakout with one lounge sofa, one coffee table. Default materials.",
    "Wrong: aligning the breakout sofa toward the dead-end wall hides participants from the meeting; right: angle the sofa toward the L's inside corner so breakout occupants can rejoin the meeting.",
    (5, 3), (3, 3), ["meeting table", "chair", "lounge sofa", "coffee table"], extra_tokens=["six"]))

TASKS.append(l_easy(28, "L-shaped garage workshop",
    "Create an L-shaped garage workshop. Long arm 6 m × 4 m holds one workbench, one tool chest; short arm 4 m × 3 m holds one parts shelving. Default materials.",
    "Wrong: pushing the tool chest into the inside corner makes drawers unreachable; right: place the tool chest flush against the long arm's outer wall with 1 m clearance for drawer pull.",
    (6, 4), (4, 3), ["workbench", "tool chest", "parts shelving"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_partition easy (8)   T0E029–T0E033 train · T0E034–T0E036 test
# ════════════════════════════════════════════════════════════════════════

def part_easy(suffix, name, prompt, distractor, dims, furniture, extra_tokens=None):
    id = f"T0E{suffix:03d}"
    split = "train" if suffix <= 33 else "test"
    w, d = dims
    grep = ["partition", "_".join(f.replace(" ", "_").replace("-", "_") for f in furniture)]
    if extra_tokens:
        grep += extra_tokens
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect overall room dims {w} m × {d} m",
        "expect exactly one interior partition wall",
        f"expect expected_furniture={furniture}",
        f"expect min_furniture={len(furniture)}",
    ]
    return task(id, "scene_indoor_partition", "easy", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(part_easy(29, "Two-zone office with full-height partition",
    "Create a 10 m × 6 m office split by one full-height interior partition wall (drywall) into two equal 5 m × 6 m zones. Each zone holds one desk and one office chair. Default RF materials.",
    "Wrong: defining the partition as a thin floating segment without joining to the perimeter walls leaves gaps at the wall junctions; right: have the partition span the full 6 m depth and join both perimeter walls.",
    (10, 6), ["desk", "office chair"], extra_tokens=["full_height"]))

TASKS.append(part_easy(30, "Classroom partitioned for two grades",
    "Create a 12 m × 6 m classroom with one interior partition (drywall) dividing it into two 6 m × 6 m zones. Each zone has one teacher desk and one student table. Default materials.",
    "Wrong: placing the teacher desks back-to-back at the partition creates direct sound bleed; right: place each teacher desk on the outer wall of its zone so the partition isolates the two groups.",
    (12, 6), ["teacher desk", "student table"]))

TASKS.append(part_easy(31, "Studio with kitchenette partition",
    "Create a 6 m × 5 m studio apartment. Place one half-height interior partition (1.2 m tall, drywall) creating a 3 m × 5 m living zone with one sofa and a 3 m × 5 m kitchenette zone with one kitchen counter. Default materials.",
    "Wrong: making the partition full-height defeats the studio's open feel and changes RF propagation; right: limit partition height to 1.2 m so air and signals couple over the top.",
    (6, 5), ["sofa", "kitchen counter"], extra_tokens=["half_height"]))

TASKS.append(part_easy(32, "Open office with cubicle wall",
    "Create a 8 m × 6 m open office with one cubicle partition (1.5 m tall, fabric panel) creating a private 3 m × 6 m alcove with one desk, one chair, and one filing cabinet; the open 5 m × 6 m area has one collaboration table. Default materials.",
    "Wrong: extending the cubicle partition all the way to the ceiling violates open-office plan; right: keep partition at 1.5 m so it provides visual privacy but preserves shared ceiling.",
    (8, 6), ["desk", "chair", "filing cabinet", "collaboration table"]))

TASKS.append(part_easy(33, "Restaurant with kitchen partition",
    "Create a 10 m × 6 m restaurant space with one full-height kitchen partition (drywall + stainless steel side) creating a 4 m × 6 m kitchen with one cooking range, one prep table, and a 6 m × 6 m dining area with one dining table and one chair. Default materials.",
    "Wrong: omitting the partition lets kitchen heat and noise bleed to diners; right: full-height partition with stainless steel cladding on the kitchen side reflects heat back into the cooking area.",
    (10, 6), ["cooking range", "prep table", "dining table", "chair"]))

TASKS.append(part_easy(34, "Lab with biosafety partition",
    "Create a 8 m × 5 m research lab with one full-height interior partition (drywall) creating a 4 m × 5 m biosafety zone with one biosafety cabinet and one lab stool, and a 4 m × 5 m general zone with one workbench. Default materials.",
    "Wrong: putting the biosafety cabinet in the general zone defeats containment; right: place the cabinet inside the partitioned biosafety zone and pass-through items via a dedicated airlock.",
    (8, 5), ["biosafety cabinet", "lab stool", "workbench"]))

TASKS.append(part_easy(35, "Home gym with mirror partition",
    "Create a 6 m × 5 m home gym with one half-height partition (1.5 m tall, mirror-fronted on the workout side) creating a 4 m × 5 m workout zone with one treadmill, one weight rack, and a 2 m × 5 m equipment storage zone with one storage shelf. Default materials.",
    "Wrong: making the partition full-height encloses the storage zone and traps moisture; right: keep partition at 1.5 m so air circulates and the mirror serves the workout zone.",
    (6, 5), ["treadmill", "weight rack", "storage shelf"]))

TASKS.append(part_easy(36, "Workshop split storage/work zones",
    "Create a 8 m × 5 m workshop with one full-height interior partition (drywall) creating a 5 m × 5 m work zone with one workbench, one tool chest, and a 3 m × 5 m storage zone with one parts shelving. Default materials.",
    "Wrong: not partitioning the storage area lets sawdust contaminate finished work; right: full-height partition with a sliding door isolates storage dust from active work.",
    (8, 5), ["workbench", "tool chest", "parts shelving"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_mixed_materials easy (10)   T0E037–T0E042 train · T0E043–T0E046 test
# ════════════════════════════════════════════════════════════════════════

def mat_easy(suffix, name, prompt, distractor, dims, furniture, materials,
             extra_tokens=None):
    id = f"T0E{suffix:03d}"
    split = "train" if suffix <= 42 else "test"
    w, d = dims
    mat_tokens = "_".join(m.replace(" ", "_").replace("-", "_") for m in materials)
    grep = [mat_tokens, "_".join(f.replace(" ", "_").replace("-", "_") for f in furniture)]
    if extra_tokens:
        grep += extra_tokens
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect overall room dims {w} m × {d} m",
        f"expect material set includes {materials}",
        f"expect at least {len(materials)} distinct RF materials",
        f"expect expected_furniture={furniture}",
    ]
    return task(id, "scene_indoor_mixed_materials", "easy", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(mat_easy(37, "Living room with wood floor and glass door",
    "Create a 5 m × 4 m living room. Floor material is hardwood, walls are drywall, one wall has a 2 m × 2.1 m glass sliding door (glass). Place one sofa, one coffee table, one TV stand. Use the three RF materials hardwood / drywall / glass.",
    "Wrong: assigning the same default material to floor, walls, and door collapses three distinct propagation surfaces into one; right: assign hardwood, drywall, and glass separately so each surface contributes its own reflection coefficient.",
    (5, 4), ["sofa", "coffee table", "tv stand"], ["hardwood", "drywall", "glass"]))

TASKS.append(mat_easy(38, "Bathroom with tile floor and ceramic walls",
    "Create a 3 m × 2.5 m bathroom. Floor material is ceramic tile, walls are ceramic tile (wet area), one mirror is mounted on a wall. Place one toilet, one sink unit, one bathtub. Use the two RF materials ceramic / mirror.",
    "Wrong: treating the mirror as a wall surface skips its specular reflection; right: declare the mirror as a separate material with metal-equivalent reflectivity.",
    (3, 2.5), ["toilet", "sink unit", "bathtub"], ["ceramic", "mirror"]))

TASKS.append(mat_easy(39, "Kitchen with quartz counter and metal appliances",
    "Create a 4 m × 3.5 m kitchen. Floor is wood, walls are drywall, counter top is quartz, one refrigerator and one oven have stainless steel fronts. Place one kitchen counter, one refrigerator, one oven, one dishwasher. Use the four RF materials wood / drywall / quartz / metal.",
    "Wrong: lumping appliances under one 'metal' tag ignores their distinct sizes for shadowing; right: each appliance gets its own object with material metal so RT tracks each one individually.",
    (4, 3.5), ["kitchen counter", "refrigerator", "oven", "dishwasher"],
    ["wood", "drywall", "quartz", "metal"]))

TASKS.append(mat_easy(40, "Office with carpet and metal door",
    "Create a 4 m × 3 m office. Floor is carpet, walls are drywall, the door is metal. Place one desk, one chair, one bookshelf. Use the three RF materials carpet / drywall / metal.",
    "Wrong: declaring the metal door inside the drywall material list collapses two surfaces; right: door object has material metal and is co-located with the perimeter wall but materially distinct.",
    (4, 3), ["desk", "chair", "bookshelf"], ["carpet", "drywall", "metal"]))

TASKS.append(mat_easy(41, "Music room with cork floor and foam panels",
    "Create a 4 m × 3 m music room. Floor is cork, walls are drywall with acoustic foam panels on two walls (foam material). Place one upright piano, one piano bench, one music stand. Use the three RF materials cork / drywall / foam.",
    "Wrong: applying foam to the entire wall surface eliminates flutter echo but also wipes out high-frequency reflections; right: foam covers ~30% of wall area, leaving drywall exposed elsewhere.",
    (4, 3), ["upright piano", "piano bench", "music stand"], ["cork", "drywall", "foam"]))

TASKS.append(mat_easy(42, "Gym with rubber floor and mirror wall",
    "Create a 5 m × 4 m home gym. Floor is rubber, walls are drywall except one wall is fully mirror (mirror material). Place one treadmill, one weight bench, one dumbbell rack. Use the three RF materials rubber / drywall / mirror.",
    "Wrong: declaring the mirror wall as drywall ignores its strong specular reflection; right: declare it as a mirror material so RT models the spec reflection lobe.",
    (5, 4), ["treadmill", "weight bench", "dumbbell rack"], ["rubber", "drywall", "mirror"]))

TASKS.append(mat_easy(43, "Photo studio with vinyl floor and blackout curtains",
    "Create a 5 m × 4 m photo studio. Floor is vinyl, walls are drywall painted white, one wall has blackout curtains (fabric material). Place one backdrop stand, one light stand, one tripod stool. Use the three RF materials vinyl / drywall / fabric.",
    "Wrong: treating curtains as drywall ignores their high-frequency absorption; right: curtains declared as fabric with low reflectivity.",
    (5, 4), ["backdrop stand", "light stand", "tripod stool"], ["vinyl", "drywall", "fabric"]))

TASKS.append(mat_easy(44, "Lab with epoxy floor and FRP walls",
    "Create a 5 m × 4 m lab. Floor is epoxy, walls are fiberglass-reinforced plastic (FRP), one wall has a glass observation window. Place one workbench, one fume hood, one lab stool. Use the three RF materials epoxy / FRP / glass.",
    "Wrong: omitting the glass window in materials makes the wall look uniform to RT; right: declare the window as a distinct glass region within the FRP wall.",
    (5, 4), ["workbench", "fume hood", "lab stool"], ["epoxy", "FRP", "glass"]))

TASKS.append(mat_easy(45, "Server closet with raised floor and metal door",
    "Create a 3 m × 3 m server closet. Floor is a raised access floor (metal panels above plenum), walls are drywall, door is metal. Place one server rack, one KVM cart, one cable tray. Use the three RF materials metal / drywall / cable.",
    "Wrong: ignoring the raised floor's metal panels misses a ground plane below the racks; right: declare the raised floor as metal so it acts as a reflector.",
    (3, 3), ["server rack", "KVM cart", "cable tray"], ["metal", "drywall", "cable"]))

TASKS.append(mat_easy(46, "Library with hardwood floor and oak shelves",
    "Create a 5 m × 4 m home library. Floor is hardwood, walls are drywall, three bookshelves are solid oak (wood material distinct from floor hardwood). Place three bookshelves, one reading chair, one side table. Use the three RF materials hardwood / drywall / wood.",
    "Wrong: collapsing hardwood floor and oak shelves into one wood material ignores their geometric difference; right: keep hardwood as a horizontal slab and oak as vertical shelving — both wood but at different surface orientations.",
    (5, 4), ["bookshelf", "reading chair", "side table"],
    ["hardwood", "drywall", "wood"], extra_tokens=["three"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_irc_compliance easy (4)   T0E047–T0E048 train · T0E049–T0E050 test
# ════════════════════════════════════════════════════════════════════════

def irc_easy(suffix, name, prompt, distractor, dims, room_type, furniture,
             extra_tokens=None):
    id = f"T0E{suffix:03d}"
    split = "train" if suffix <= 48 else "test"
    w, d = dims
    floor = w * d
    grep = ["irc", "window", "_".join(f.replace(" ", "_").replace("-", "_") for f in furniture)]
    if extra_tokens:
        grep += extra_tokens
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect single habitable room of type '{room_type}'",
        f"expect floor area {floor:.1f} m^2",
        f"expect total window aperture >= {0.08*floor:.2f} m^2 (8% of floor)",
        "expect all windows on perimeter walls (north/south/east/west)",
        f"expect expected_furniture={furniture}",
    ]
    return task(id, "scene_indoor_irc_compliance", "easy", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(irc_easy(47, "IRC-compliant bedroom",
    "Create a 4 m × 4 m bedroom (room_type='bedroom'). Add one window on a perimeter wall, sized so its area is at least 8% of the floor area (1.28 m²); 1.2 m × 1.2 m suffices. Place one double bed, one nightstand, one wardrobe. Default RF materials.",
    "Wrong: placing the window on the interior wall (no exterior view) violates IRC §R303 perimeter requirement; right: window goes on north/south/east/west perimeter wall.",
    (4, 4), "bedroom", ["double bed", "nightstand", "wardrobe"]))

TASKS.append(irc_easy(48, "IRC-compliant living room (two windows)",
    "Create a 5 m × 4 m living room (room_type='living'). Add two windows on perimeter walls totaling at least 8% of the floor area (1.60 m²); two 1.2 m × 1.0 m windows on adjacent perimeter walls work. Place one sofa, one coffee table, one tv stand. Default materials.",
    "Wrong: splitting one large window across an interior corner makes its aperture invalid; right: distribute the two windows across two perimeter walls so both count.",
    (5, 4), "living", ["sofa", "coffee table", "tv stand"], extra_tokens=["two"]))

TASKS.append(irc_easy(49, "IRC-compliant kitchen",
    "Create a 4 m × 3 m kitchen (room_type='kitchen'). Add one window on a perimeter wall sized to meet 8% of floor area (0.96 m²). Place one kitchen counter, one refrigerator, one sink unit. Default materials.",
    "Wrong: using an interior pass-through as a 'window' fails IRC because it doesn't reach the exterior; right: window on perimeter wall facing outdoors.",
    (4, 3), "kitchen", ["kitchen counter", "refrigerator", "sink unit"]))

TASKS.append(irc_easy(50, "IRC-compliant home office",
    "Create a 4 m × 3 m home office (room_type='office'). Add one window on a perimeter wall sized to meet 8% of floor area (0.96 m²). Place one desk, one office chair, one bookshelf. Default materials.",
    "Wrong: placing a skylight as the only opening — IRC requires perimeter window aperture distinct from roof; right: add a wall-mounted window on a perimeter wall.",
    (4, 3), "office", ["desk", "office chair", "bookshelf"]))


# ════════════════════════════════════════════════════════════════════════
# scene_edit easy (10)   T0E051–T0E056 train · T0E057–T0E060 test
# ════════════════════════════════════════════════════════════════════════

FLOORPLANS = {
    "apartment": "benchmark/scenes/floorplans/apartment/scene_state.json",
    "office":    "benchmark/scenes/floorplans/office/scene_state.json",
    "warehouse": "benchmark/scenes/floorplans/warehouse/scene_state.json",
}


def edit_easy(suffix, name, prompt, distractor, source_key, assertions_extra,
              grep_tokens):
    id = f"T0E{suffix:03d}"
    split = "train" if suffix <= 56 else "test"
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect input scene loaded from {FLOORPLANS[source_key]}",
        "expect output scene_state.json reflects the requested edit",
        "expect non-edited objects preserved unchanged",
    ] + assertions_extra
    grep = ["scene_edit", source_key] + grep_tokens
    return {
        **task(id, "scene_edit", "easy", split, name, prompt, distractor,
               assertions, grep),
        "scene_path": FLOORPLANS[source_key],
    }


TASKS.append(edit_easy(51, "Move the sofa (apartment)",
    "Take the pre-shipped apartment scene_state.json. Move the sofa from its current position to be flush against the opposite living-room wall. Preserve all other objects and materials.",
    "Wrong: writing a fresh scene_state.json from scratch loses the apartment's existing room polygons and other furniture; right: load the pre-shipped JSON, mutate only the sofa's position fields, and re-dump.",
    "apartment", ["expect sofa moved", "expect sofa orientation preserved"],
    ["sofa", "moved"]))

TASKS.append(edit_easy(52, "Change desk material to glass (office)",
    "Take the pre-shipped office scene_state.json. Change the material attribute of the desk from its current value to 'glass'. Preserve geometry and all other objects.",
    "Wrong: renaming the desk also changes its id, breaking references elsewhere; right: keep id and geometry, modify only the material field.",
    "office", ["expect desk.material == 'glass'", "expect desk id unchanged"],
    ["desk", "glass", "material"]))

TASKS.append(edit_easy(53, "Add a workbench (warehouse)",
    "Take the pre-shipped warehouse scene_state.json. Add one workbench (2 m × 0.7 m) flush against an unused interior wall. Default material wood. Preserve existing objects.",
    "Wrong: placing the new workbench at coordinates that overlap existing shelving fails collision check; right: query bounding boxes of existing objects, then place the workbench in a free wall segment.",
    "warehouse", ["expect new workbench in scene", "expect collision-free addition"],
    ["workbench", "added"]))

TASKS.append(edit_easy(54, "Remove the coffee table (apartment)",
    "Take the pre-shipped apartment scene_state.json. Remove the coffee table object entirely. Preserve all other objects and their positions.",
    "Wrong: deleting the coffee table id but keeping a dangling reference in another object's metadata breaks scene integrity; right: remove the object record and any references, then dump.",
    "apartment", ["expect coffee table removed", "expect no dangling references"],
    ["coffee", "table", "removed"]))

TASKS.append(edit_easy(55, "Add a meeting table (office)",
    "Take the pre-shipped office scene_state.json. Add one meeting table (2.4 m × 1.2 m) centered in the largest open zone. Default material wood. Preserve existing objects.",
    "Wrong: placing the meeting table at the geometric centroid without checking existing furniture causes overlap; right: compute the largest empty rectangle and place the meeting table inside it with 0.6 m chair clearance.",
    "office", ["expect new meeting table in scene", "expect collision-free addition"],
    ["meeting", "table", "added"]))

TASKS.append(edit_easy(56, "Change floor material to epoxy (warehouse)",
    "Take the pre-shipped warehouse scene_state.json. Change the floor material from its current value to 'epoxy'. Preserve all object positions and other materials.",
    "Wrong: changing every wood-tagged surface to epoxy also overwrites wood pallets; right: scope the change to the floor object only, leaving wood furniture/pallets unchanged.",
    "warehouse", ["expect floor.material == 'epoxy'", "expect non-floor wood preserved"],
    ["floor", "epoxy", "material"]))

# scene_edit easy test (T0E057–T0E060)
TASKS.append(edit_easy(57, "Swap TV stand for media console (apartment)",
    "Take the pre-shipped apartment scene_state.json. Replace the TV stand with a media console (1.8 m × 0.4 m, material wood) at the same wall position. Preserve all other objects.",
    "Wrong: deleting the TV stand id without re-using the wall slot leaves a gap; right: keep the wall position, change only object kind and dimensions.",
    "apartment", ["expect tv stand removed", "expect media console added at same wall slot"],
    ["media", "console", "swapped"]))

TASKS.append(edit_easy(58, "Rotate the desk 90 degrees (office)",
    "Take the pre-shipped office scene_state.json. Rotate the desk by 90 degrees around its centroid. Preserve position, material, and all other objects.",
    "Wrong: rotating by 90° but keeping the original w/d swapped in the wrong field produces a footprint inconsistent with the rotation; right: set theta=90 and either swap width/depth or use the rotated AABB for collision check.",
    "office", ["expect desk.theta == 90", "expect desk centroid unchanged"],
    ["desk", "rotated", "ninety"]))

TASKS.append(edit_easy(59, "Add two storage shelves (warehouse)",
    "Take the pre-shipped warehouse scene_state.json. Add two storage shelves (each 2 m × 0.5 m, material metal) along an unused wall, parallel to each other with 0.8 m aisle clearance. Preserve existing objects.",
    "Wrong: stacking the two shelves at the same wall coordinate produces collision; right: offset them by 2 m along the wall so each gets its own footprint.",
    "warehouse", ["expect two new storage shelves in scene", "expect 0.8 m aisle clearance between them"],
    ["storage", "shelves", "two"]))

TASKS.append(edit_easy(60, "Change exterior walls to brick (apartment)",
    "Take the pre-shipped apartment scene_state.json. Change all exterior wall materials to 'brick'. Preserve interior wall materials and all furniture.",
    "Wrong: rewriting the materials list as a single string overwrites interior wall materials too; right: filter walls by 'is_exterior=true' or boundary tag, then change only those.",
    "apartment", ["expect exterior wall material == 'brick'", "expect interior wall materials unchanged"],
    ["exterior", "brick", "wall"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_l_shape hard (6)   T0H001–T0H003 train · T0H004–T0H006 test
# ════════════════════════════════════════════════════════════════════════

def l_hard(suffix, name, prompt, distractor, assertions_extra, grep_extra):
    id = f"T0H{suffix:03d}"
    split = "train" if suffix <= 3 else "test"
    grep = ["l_shape"] + grep_extra
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
    ] + assertions_extra
    return task(id, "scene_indoor_l_shape", "hard", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(l_hard(1, "L-shaped open office (parametric area)",
    "Generate an L-shaped open office. Long arm exactly 8 m × 4 m, short arm exactly 4 m × 4 m, joined at the inside corner so total area is 48 m². Place exactly 4 desks (1.4 m × 0.7 m each) distributed: 3 along the long arm's outer wall and 1 in the short arm, each desk paired with one office chair. Walls drywall throughout.",
    "Wrong: rounding to 50 m² loses the exact L-fit, leaving 2 m² of unallocated space; right: keep both arms at their exact dimensions so total polygon area is 48 m² and the partition between arms is shared (not double-counted).",
    [
        "expect long arm 8 m × 4 m",
        "expect short arm 4 m × 4 m",
        "expect total floor area 48 m^2 (±0.5)",
        "expect 4 desks and 4 office chairs",
        "expect drywall walls throughout",
    ],
    ["eight", "four", "drywall", "desk", "chair", "open_office"]))

TASKS.append(l_hard(2, "L-shaped workshop with mixed materials",
    "Generate an L-shaped workshop. Long arm 6 m × 4 m is the active work area with concrete floor (industrial wear) and drywall walls; short arm 4 m × 3 m is storage with epoxy floor and metal-clad walls. Place one workbench and one tool chest in the work arm, one parts shelving in the storage arm. Door between arms is metal.",
    "Wrong: assigning concrete to the full L floor ignores the storage arm's epoxy spec; right: split the floor object at the inside corner so each arm holds its own material.",
    [
        "expect work arm 6 m × 4 m with concrete floor and drywall walls",
        "expect storage arm 4 m × 3 m with epoxy floor and metal-clad walls",
        "expect connecting door material metal",
        "expect floors split by interior corner so each arm has its own material",
    ],
    ["concrete", "epoxy", "drywall", "metal", "workbench"]))

TASKS.append(l_hard(3, "L-shaped apartment (kitchen + living)",
    "Generate an L-shaped apartment unit. Long arm 7 m × 4 m living area with one sofa, one coffee table, one tv stand; short arm 4 m × 4 m kitchen with one kitchen counter, one refrigerator, one oven, one dining table for 2. Inside corner stays open for sightline; exterior walls concrete; interior surfaces drywall; kitchen floor tile, living floor hardwood.",
    "Wrong: walling off the inside corner kills the sightline that defines L-shaped open living; right: leave the corner open so kitchen and living share visual flow, but split floor materials precisely at the corner.",
    [
        "expect long arm 7 m × 4 m (living)",
        "expect short arm 4 m × 4 m (kitchen)",
        "expect open inside corner (no wall)",
        "expect exterior walls concrete, interior surfaces drywall",
        "expect kitchen floor tile, living floor hardwood",
        "expect 4 living furniture items and 4 kitchen items",
    ],
    ["concrete", "drywall", "tile", "hardwood", "sofa", "kitchen_counter"]))

# l_shape hard test
TASKS.append(l_hard(4, "L-shaped medical clinic",
    "Generate an L-shaped clinic suite. Long arm 6 m × 4 m is the waiting area with 4 chairs and one reception desk; short arm 4 m × 3 m is the exam room with one exam table and one doctor stool. Door at the inside corner separates the two. Walls drywall; exam room floor vinyl; waiting floor carpet.",
    "Wrong: leaving the inside corner open between waiting and exam breaks patient privacy; right: place a door at the inside corner so the exam room is isolated.",
    [
        "expect waiting arm 6 m × 4 m, exam arm 4 m × 3 m",
        "expect door at inside corner",
        "expect 4 chairs + reception desk in waiting",
        "expect exam table + doctor stool in exam",
        "expect waiting floor carpet, exam floor vinyl",
    ],
    ["waiting", "exam", "carpet", "vinyl", "door"]))

TASKS.append(l_hard(5, "L-shaped photography studio",
    "Generate an L-shaped photography studio. Long arm 7 m × 4 m is the shooting zone with one backdrop stand, two light stands, one tripod stool; short arm 4 m × 3 m is the editing zone with one editing desk, one monitor stand, one editing chair. Inside corner has a heavy curtain (fabric material) instead of a hard wall.",
    "Wrong: using a hard wall at the inside corner blocks photographer movement; right: install a curtain on a ceiling track so the corner can open/close for staging.",
    [
        "expect shoot arm 7 m × 4 m, edit arm 4 m × 3 m",
        "expect curtain (fabric) at inside corner instead of hard wall",
        "expect 4 shoot items and 3 edit items",
    ],
    ["backdrop", "editing", "curtain", "fabric"]))

TASKS.append(l_hard(6, "L-shaped preschool classroom",
    "Generate an L-shaped preschool classroom. Long arm 6 m × 5 m is the learning zone with one teacher desk, 4 child tables, 8 child chairs; short arm 4 m × 3 m is the nap zone with 6 floor mats and one storage shelf. Soft separation via a low (0.9 m) bookshelf along the inside corner instead of a full wall.",
    "Wrong: full-height wall between zones reduces teacher line-of-sight to nappers; right: use a 0.9 m bookshelf at the inside corner so the teacher can still see across both zones.",
    [
        "expect learn arm 6 m × 5 m, nap arm 4 m × 3 m",
        "expect 0.9 m bookshelf at inside corner",
        "expect 4 child tables + 8 child chairs in learn zone",
        "expect 6 floor mats in nap zone",
    ],
    ["learning", "nap", "bookshelf", "child_table", "floor_mat"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_partition hard (6)   T0H007–T0H010 train · T0H011–T0H012 test
# ════════════════════════════════════════════════════════════════════════

def part_hard(suffix, name, prompt, distractor, assertions_extra, grep_extra):
    id = f"T0H{suffix:03d}"
    split = "train" if suffix <= 10 else "test"
    grep = ["partition"] + grep_extra
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
    ] + assertions_extra
    return task(id, "scene_indoor_partition", "hard", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(part_hard(7, "Lab with glass partition (clean/dirty zones)",
    "Generate a 10 m × 6 m lab. Place one full-height glass partition (2.4 m tall, 6 m long) running parallel to the short walls, splitting the lab into a 4 m × 6 m clean zone with one biosafety cabinet and one lab stool, and a 6 m × 6 m dirty zone with one workbench, one fume hood, and one lab stool. A 0.9 m metal-framed glass door at the centre of the partition. Clean zone floor is epoxy; dirty zone floor is sealed concrete. Walls drywall.",
    "Wrong: using drywall for the partition kills the visual sightline that's critical for supervision between zones; right: use glass for the partition body and a metal frame around the door so RT models both materials.",
    [
        "expect overall dims 10 m × 6 m",
        "expect glass partition 6 m long, 2.4 m tall",
        "expect clean zone 4 m × 6 m with epoxy floor",
        "expect dirty zone 6 m × 6 m with sealed concrete floor",
        "expect 0.9 m glass door with metal frame at partition centre",
    ],
    ["glass", "epoxy", "concrete", "drywall", "biosafety_cabinet", "fume_hood"]))

TASKS.append(part_hard(8, "Two-zone office with half-height partition (different floors)",
    "Generate a 8 m × 6 m office split by one half-height partition (1.2 m tall, fabric panel, 6 m long) into a 4 m × 6 m collaboration zone with one collaboration table, 4 chairs, and a 4 m × 6 m focus zone with 4 individual desks, 4 office chairs. Collaboration floor is carpet; focus floor is cork. Walls drywall throughout.",
    "Wrong: making the partition full-height removes the open feel; right: half-height fabric panel at 1.2 m provides visual privacy at sitting height while preserving open ceiling and natural light.",
    [
        "expect overall 8 m × 6 m",
        "expect partition 1.2 m tall, fabric, 6 m long",
        "expect collab floor carpet, focus floor cork",
        "expect 4 collab chairs + 4 focus desks + 4 office chairs",
    ],
    ["half_height", "fabric", "carpet", "cork", "collaboration", "focus"]))

TASKS.append(part_hard(9, "Classroom + AV control room (acoustic)",
    "Generate a 12 m × 6 m space. Place one full-height partition with double drywall + insulation (2.4 m tall, 6 m long) separating a 10 m × 6 m classroom (one teacher desk, 8 student tables, 16 student chairs, projector screen on one wall) from a 2 m × 6 m AV control room (one AV console, one operator chair, one server rack). One 0.9 m soundproof door at the partition. Classroom carpet floor; AV room vinyl floor.",
    "Wrong: single drywall partition lets HVAC and AV system noise bleed; right: double drywall with insulation core gives ~50 dB transmission loss needed for AV control.",
    [
        "expect overall 12 m × 6 m, classroom 10 m × 6 m, AV room 2 m × 6 m",
        "expect double-drywall partition with insulation core",
        "expect soundproof door (0.9 m wide)",
        "expect 8 student tables and 16 student chairs in classroom",
        "expect AV console + operator chair + server rack in control room",
    ],
    ["double_drywall", "insulation", "soundproof", "classroom", "av_console"]))

TASKS.append(part_hard(10, "Restaurant kitchen partition (stainless side)",
    "Generate a 12 m × 6 m restaurant. Place one full-height kitchen partition (drywall body with stainless steel cladding on the kitchen side, 2.4 m tall, 6 m long) splitting into a 5 m × 6 m kitchen (one cooking range, one prep table, one walk-in cold room) and a 7 m × 6 m dining area (4 dining tables, 16 chairs, one host stand). One double swing door at the partition.",
    "Wrong: stainless on the dining side reflects heat back to diners; right: stainless on the kitchen side, drywall painted on dining side — material asymmetry preserves both safety and aesthetics.",
    [
        "expect kitchen 5 m × 6 m, dining 7 m × 6 m",
        "expect asymmetric partition (stainless kitchen-side, drywall dining-side)",
        "expect double swing door at partition centre",
        "expect 4 dining tables + 16 chairs in dining",
    ],
    ["stainless", "drywall", "swing", "asymmetric", "cooking_range"]))

# partition hard test
TASKS.append(part_hard(11, "Hospital ward with curtain rails",
    "Generate a 10 m × 6 m hospital ward. Place one half-height partition (1.5 m tall, drywall) and three ceiling-mounted curtain rails (each 3 m long) dividing the room into four 2.5 m × 6 m patient bays. Each bay has one hospital bed, one bedside table, one IV stand. Floor vinyl throughout; walls drywall.",
    "Wrong: full-height walls for each bay violate ward visibility for nursing staff; right: 1.5 m partitions with ceiling-mounted curtain rails preserve nurse line-of-sight from corridor while giving patient privacy.",
    [
        "expect 4 patient bays of 2.5 m × 6 m each",
        "expect 3 ceiling-mounted curtain rails (each 3 m)",
        "expect each bay: 1 hospital bed + 1 bedside table + 1 IV stand",
    ],
    ["curtain_rail", "patient_bay", "hospital_bed", "iv_stand"]))

TASKS.append(part_hard(12, "Library + study room (glass partition)",
    "Generate a 10 m × 6 m library. Place one full-height glass partition (2.4 m tall, 4 m long) creating a 4 m × 4 m enclosed quiet study room with 2 study tables, 4 chairs, while the remaining 8 m × 6 m + 6 m × 2 m L-shaped open library area holds 6 bookshelves, 2 reading chairs, 2 side tables. Glass door at the partition.",
    "Wrong: drywall partition blocks light and natural surveillance into the study room; right: glass partition keeps study quiet while preserving librarian sightline.",
    [
        "expect study room 4 m × 4 m (full glass partition)",
        "expect open area is the remaining L-shaped polygon",
        "expect 2 study tables + 4 chairs in study room",
        "expect 6 bookshelves + 2 reading chairs in open area",
    ],
    ["glass", "study_room", "bookshelf", "reading_chair"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_mixed_materials hard (6)   T0H013–T0H015 train · T0H016–T0H018 test
# ════════════════════════════════════════════════════════════════════════

def mat_hard(suffix, name, prompt, distractor, assertions_extra, grep_extra):
    id = f"T0H{suffix:03d}"
    split = "train" if suffix <= 15 else "test"
    grep = ["mixed_materials"] + grep_extra
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
    ] + assertions_extra
    return task(id, "scene_indoor_mixed_materials", "hard", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(mat_hard(13, "Server room with 60 GHz transmission constraints",
    "Generate a 8 m × 6 m server room. Concrete walls (0.25 m thick), drywall ceiling, raised metal access floor with cable plenum below. Place 4 server racks (each 0.6 m × 0.8 m, metal) in a hot/cold aisle layout, one KVM cart, one cable tray spanning the ceiling. Targeted for 60 GHz mmWave AP placement so material choice must reflect that band: concrete walls (~30 dB/cm at 60 GHz), metal rack faces (specular reflection).",
    "Wrong: assigning generic 'wall' material ignores the 60 GHz penetration loss difference between concrete (~30 dB/cm) and drywall (~3 dB/cm); right: each surface gets the frequency-aware material so RT picks the right reflection/transmission coefficient.",
    [
        "expect concrete walls 0.25 m thick, drywall ceiling",
        "expect raised metal access floor",
        "expect 4 metal server racks in hot/cold aisle",
        "expect at least 4 distinct RF materials: concrete, drywall, metal, plenum",
        "expect 60 GHz frequency-aware material assignment",
    ],
    ["concrete", "drywall", "metal", "plenum", "60ghz", "server_rack"]))

TASKS.append(mat_hard(14, "Recording studio with bass traps and diffuser",
    "Generate a 6 m × 5 m recording studio. Floating floor (rubber on isolators), drywall walls with bass traps (foam-filled) in all 4 corners, diffuser panels (wood) on the rear wall, fabric absorbers on first reflection points on side walls. One glass control window (1.2 m × 1 m) on one wall facing the control room. Place one drum kit, one vocal booth, one mic stand, one acoustic guitar stand.",
    "Wrong: treating all walls with the same absorber kills room reverb entirely; right: balance bass traps (corner foam), mid absorbers (fabric on side walls), and diffusers (wood at rear) so the room sounds live but controlled.",
    [
        "expect floating rubber floor",
        "expect bass traps (foam) in all 4 corners",
        "expect diffuser panels (wood) on rear wall",
        "expect fabric absorbers on first reflection points",
        "expect glass control window 1.2 m × 1 m",
        "expect at least 5 distinct materials",
    ],
    ["floating_floor", "bass_trap", "diffuser", "fabric", "glass"]))

TASKS.append(mat_hard(15, "Cleanroom with HEPA ceiling and airlock",
    "Generate a 6 m × 5 m ISO Class 7 cleanroom. FRP walls, epoxy floor with coved base, HEPA filtered ceiling (filter material plus aluminum grid), one airlock door (4-step interlock: outer door, gowning vestibule 1.5 m × 1 m, inner door). Place one biosafety cabinet, one wet bench, one drying oven, one lab stool.",
    "Wrong: skipping the gowning vestibule between the outer and inner airlock doors breaks Class 7 contamination control; right: model the vestibule as a separate 1.5 m × 1 m sub-room with both doors closed during transit.",
    [
        "expect FRP walls, epoxy floor with coved base",
        "expect HEPA ceiling (filter + aluminum grid)",
        "expect 4-step airlock (outer door + 1.5 m × 1 m gowning vestibule + inner door)",
        "expect biosafety cabinet + wet bench + drying oven + lab stool",
        "expect at least 5 distinct materials",
    ],
    ["frp", "epoxy", "hepa", "aluminum", "airlock", "gowning"]))

# mixed_materials hard test
TASKS.append(mat_hard(16, "Hospital MRI room with RF shielding",
    "Generate a 7 m × 5 m hospital MRI room. Walls 1 m of solid copper shielding (Faraday cage) on all sides and ceiling, lead-lined door (0.6 m wide), one RF-protected window (copper mesh inside glass) for radiographer line-of-sight. Floor is non-magnetic vinyl. Place one MRI scanner (3 m × 1.5 m), one patient table, one anesthesia trolley (non-ferrous).",
    "Wrong: any ferromagnetic material inside the MRI room (e.g., steel-frame furniture) is dangerous and breaks RT material assumptions; right: every furniture item explicitly tagged non-ferrous; walls are copper shielding with measurable attenuation in the 64–128 MHz Larmor band.",
    [
        "expect copper-shielded walls (Faraday cage)",
        "expect lead-lined door",
        "expect RF-protected window (copper mesh in glass)",
        "expect non-magnetic vinyl floor",
        "expect all furniture marked non-ferrous",
    ],
    ["copper", "faraday", "lead", "vinyl", "non_ferrous", "mri"]))

TASKS.append(mat_hard(17, "Industrial kitchen with grease ducting",
    "Generate a 10 m × 6 m commercial kitchen. Stainless steel cladding on all walls below 2 m, drywall above. Non-slip ceramic tile floor with floor drains. Stainless steel hood (3 m × 1 m) over the cooking range with grease ducting penetrating the ceiling. Place one cooking range, one prep table, one walk-in freezer, one dishwasher.",
    "Wrong: putting the hood without ducting penetration in the ceiling traps grease vapor; right: ducting through the ceiling creates a circular metal aperture distinct from the drywall ceiling material.",
    [
        "expect stainless cladding below 2 m, drywall above",
        "expect non-slip ceramic tile floor with drains",
        "expect stainless hood with ceiling ducting penetration",
        "expect cooking range + prep table + walk-in freezer + dishwasher",
    ],
    ["stainless", "drywall", "ceramic", "non_slip", "hood", "ducting"]))

TASKS.append(mat_hard(18, "Concert hall foyer with marble and glass curtain wall",
    "Generate a 12 m × 8 m concert hall foyer. Floor polished marble, columns marble, one 12 m × 4 m glass curtain wall on the entry side (laminated safety glass with steel mullions), oak panel walls on the interior side, drywall ceiling. Place one ticket booth (3 m × 1 m, wood), one cloakroom counter (2 m × 0.6 m), 2 standing bar tables, 6 high stools.",
    "Wrong: treating the glass curtain wall as a single uniform glass surface misses the steel mullion grid; right: model the curtain wall as a glass field with steel mullion strips at 2 m centers, each contributing distinct reflection.",
    [
        "expect polished marble floor and columns",
        "expect 12 m × 4 m glass curtain wall with steel mullions",
        "expect oak panel interior walls, drywall ceiling",
        "expect at least 5 distinct materials",
    ],
    ["marble", "glass", "steel", "oak", "drywall", "curtain_wall"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_irc_compliance hard (8)   T0H019–T0H023 train · T0H024–T0H026 test
# ════════════════════════════════════════════════════════════════════════

def irc_hard(suffix, name, prompt, distractor, assertions_extra, grep_extra):
    id = f"T0H{suffix:03d}"
    split = "train" if suffix <= 23 else "test"
    grep = ["irc", "window"] + grep_extra
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
    ] + assertions_extra
    return task(id, "scene_indoor_irc_compliance", "hard", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(irc_hard(19, "Two-bedroom apartment IRC compliance",
    "Generate a 10 m × 7 m two-bedroom apartment. Bedroom A 4 m × 4 m (room_type='bedroom'), bedroom B 3 m × 4 m (room_type='bedroom'), living 4 m × 3 m (room_type='living'), kitchen 3 m × 3 m (room_type='kitchen'). Each habitable room has at least one window on a perimeter wall meeting IRC §R303 8% aperture (bedroom A ≥ 1.28 m², bedroom B ≥ 0.96 m², living ≥ 0.96 m², kitchen ≥ 0.72 m²). Drywall interior walls 0.15 m thick, concrete exterior 0.25 m.",
    "Wrong: clustering all four windows on one façade leaves the other rooms windowless and IRC-non-compliant; right: distribute windows on perimeter walls so every habitable room has its own.",
    [
        "expect 4 rooms: 2 bedrooms, 1 living, 1 kitchen",
        "expect each habitable room has window >= 8% of its floor area",
        "expect all windows on perimeter walls",
        "expect drywall 0.15 m interior, concrete 0.25 m exterior",
    ],
    ["bedroom", "living", "kitchen", "perimeter", "drywall", "concrete"]))

TASKS.append(irc_hard(20, "Apartment with kitchen ventilation",
    "Generate a 9 m × 6 m apartment with kitchen 4 m × 4 m, living 5 m × 4 m, bedroom 4 m × 2 m. Kitchen meets both IRC §R303 8% window aperture AND requires a ventilation opening (range hood ducted to exterior, ducting 0.15 m diameter through exterior wall). Living meets 8% via windows. Bedroom meets 8% AND has an egress window (clear opening ≥ 0.6 m × 0.6 m, sill ≤ 1.1 m from floor).",
    "Wrong: ducting the range hood through an interior wall doesn't satisfy IRC kitchen ventilation; right: ducting must terminate at exterior with a 0.15 m circular aperture.",
    [
        "expect kitchen 4 m × 4 m with 8% window + exterior-ducted hood",
        "expect living 5 m × 4 m with 8% window",
        "expect bedroom 4 m × 2 m with 8% window + egress window (>=0.6 m × 0.6 m clear, sill <= 1.1 m)",
    ],
    ["egress", "ventilation", "duct", "perimeter"]))

TASKS.append(irc_hard(21, "Studio apartment compound IRC",
    "Generate a 6 m × 5 m studio apartment (room_type='living', single open zone) with one window meeting compound IRC: aperture ≥ 8% of floor area (≥ 2.4 m²), AND minimum operable area ≥ 4% of floor (≥ 1.2 m² that opens), AND emergency egress sized ≥ 0.6 m × 0.6 m clear opening with sill ≤ 1.1 m. Place one bed, one kitchenette counter, one sofa, one dining table.",
    "Wrong: a single fixed glass wall meets 8% aperture but fails 4% operable requirement; right: use a casement or double-hung window where ≥ 50% of glass area opens.",
    [
        "expect single open room 6 m × 5 m, room_type='living'",
        "expect window aperture >= 2.4 m^2 (8% of 30)",
        "expect operable area >= 1.2 m^2 (4% of 30)",
        "expect egress >= 0.6 m × 0.6 m clear opening, sill <= 1.1 m",
    ],
    ["aperture", "operable", "egress", "sill"]))

TASKS.append(irc_hard(22, "Multi-room IRC with egress windows",
    "Generate a 12 m × 8 m three-bedroom unit. Each bedroom (room_type='bedroom', each 4 m × 3 m) requires both IRC §R303 8% window aperture (≥ 0.96 m²) AND IRC §R310 egress window (clear opening ≥ 0.62 m × 0.51 m / 0.317 m² minimum, sill ≤ 1.118 m, openable). Living 4 m × 5 m needs only 8% aperture. Kitchen 4 m × 3 m needs 8% + exterior-ducted vent.",
    "Wrong: a 0.6 m × 0.5 m window meets the 8% aperture but fails the egress 0.62 m × 0.51 m minimum dimension test; right: each bedroom window sized to clear ≥ 0.62 m × 0.51 m and openable.",
    [
        "expect 3 bedrooms each 4 m × 3 m with both 8% aperture and IRC §R310 egress",
        "expect living 4 m × 5 m with 8% aperture",
        "expect kitchen 4 m × 3 m with 8% aperture + exterior-ducted vent",
    ],
    ["bedroom", "egress", "aperture", "operable"]))

TASKS.append(irc_hard(23, "Loft conversion with windows on two walls",
    "Generate a 10 m × 6 m loft converted to a residential unit (room_type='living', single zone). Two perimeter walls have windows: the long wall has 3 windows (each 1.5 m × 1.2 m), the short wall has 1 window (1.5 m × 1.2 m). Total aperture must meet IRC 8% (4.8 m² needed; planned 7.2 m²). One window on each perimeter wall must meet egress. Place one bed, one sofa, one dining table, one kitchenette counter.",
    "Wrong: a clerestory-only window arrangement (windows all above 1.5 m sill) fails egress sill requirement; right: at least one window per habitable area has its sill below 1.118 m.",
    [
        "expect single open loft 10 m × 6 m, room_type='living'",
        "expect 3 windows on long wall (each 1.5 m × 1.2 m)",
        "expect 1 window on short wall (1.5 m × 1.2 m)",
        "expect total aperture >= 4.8 m^2",
        "expect at least one window with sill <= 1.118 m and >= 0.62 m × 0.51 m clear opening",
    ],
    ["loft", "clerestory", "sill", "egress"]))

# irc hard test
TASKS.append(irc_hard(24, "Three-bedroom IRC with corridor exempt",
    "Generate a 12 m × 8 m three-bedroom unit. Each bedroom 3 m × 4 m (room_type='bedroom') has its own perimeter window meeting both 8% aperture and §R310 egress. A 12 m × 2 m corridor (room_type='corridor', non-habitable) has no window requirement. Living 4 m × 4 m and kitchen 4 m × 4 m each meet 8%.",
    "Wrong: placing a 'window' on the corridor's interior wall and counting it for an adjacent bedroom violates IRC perimeter rule; right: each bedroom's window opens directly to the exterior on its own perimeter wall.",
    [
        "expect 3 bedrooms each 3 m × 4 m with own perimeter window (8% + egress)",
        "expect corridor 12 m × 2 m (non-habitable) exempt from window requirement",
        "expect living and kitchen 4 m × 4 m each meet 8% aperture",
    ],
    ["bedroom", "corridor", "non_habitable", "perimeter"]))

TASKS.append(irc_hard(25, "Townhouse IRC (bedroom + study + kitchen)",
    "Generate a 8 m × 7 m townhouse floor unit. Bedroom 4 m × 4 m (habitable, needs 8% + egress), study 3 m × 3 m (room_type='study', habitable, needs 8% aperture only), kitchen 4 m × 3 m (room_type='kitchen', needs 8% + exterior vent). Place a bed, desk, kitchen counter, and refrigerator appropriately.",
    "Wrong: counting a study as non-habitable to skip its window requirement violates IRC; right: study is habitable, needs 8% aperture but is exempt from egress because it is not a sleeping room.",
    [
        "expect bedroom 4 m × 4 m with 8% + egress",
        "expect study 3 m × 3 m with 8% only (no egress required)",
        "expect kitchen 4 m × 3 m with 8% + exterior vent",
    ],
    ["bedroom", "study", "kitchen", "egress"]))

TASKS.append(irc_hard(26, "Garage conversion to ADU",
    "Generate a 6 m × 6 m garage converted to an Accessory Dwelling Unit (single zone, room_type='bedroom'). Existing garage door opening (3 m × 2.1 m) must be partially infilled and replaced with one IRC-compliant window (≥ 8% of 36 m² = ≥ 2.88 m², plus egress) and one residential door (0.9 m × 2.1 m). Kitchen counter and bed inside.",
    "Wrong: keeping the full garage door as 'window equivalent' fails because garage doors aren't IRC operable apertures; right: infill the garage door opening, install a 1.5 m × 2 m egress window plus a 0.9 m residential door.",
    [
        "expect single zone 6 m × 6 m, room_type='bedroom'",
        "expect garage door infilled",
        "expect 1.5 m × 2 m egress window (>=8% aperture + §R310 egress)",
        "expect 0.9 m × 2.1 m residential door",
    ],
    ["adu", "garage", "infill", "egress"]))


# ════════════════════════════════════════════════════════════════════════
# scene_indoor_multi_room hard (10)   T0H027–T0H032 train · T0H033–T0H036 test
# ════════════════════════════════════════════════════════════════════════

def mr_hard(suffix, name, prompt, distractor, assertions_extra, grep_extra):
    id = f"T0H{suffix:03d}"
    split = "train" if suffix <= 32 else "test"
    grep = ["multi_room"] + grep_extra
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
    ] + assertions_extra
    return task(id, "scene_indoor_multi_room", "hard", split, name, prompt,
                distractor, assertions, grep)


TASKS.append(mr_hard(27, "Three-bedroom apartment with central corridor",
    "Generate a 12 m × 9 m apartment containing three identical 3 m × 3 m bedrooms arranged side-by-side along a 12 m × 2 m corridor. The bedrooms open onto the corridor. Each bedroom contains one single bed (1 m × 2 m) and one wardrobe (1 m × 0.6 m); the corridor has no furniture. Interior walls 0.15 m thick drywall; exterior walls 0.25 m thick concrete.",
    "Wrong: modeling the corridor as another bedroom-sized room produces overlapping room polygons, or defining shared walls twice (once per adjacent room) double-counts wall material in propagation. Right: define exactly 4 non-overlapping room polygons (3 bedrooms + 1 corridor), declare each interior wall once with `between=[room_a, room_b]`, and tag each wall's `material` field explicitly so propagation references a single wall instance.",
    [
        "expect num_rooms=4",
        "expect 3 bedrooms each 3 m × 3 m + 1 corridor 12 m × 2 m",
        "expect each bedroom contains bed and wardrobe",
        "expect corridor furniture count = 0",
        "expect interior_wall_thickness=0.15, material drywall",
        "expect exterior_wall_thickness=0.25, material concrete",
        "expect total floor area in [100, 116] m^2",
    ],
    ["three", "bedroom", "corridor", "drywall", "concrete", "bed_wardrobe"]))

TASKS.append(mr_hard(28, "Two-bedroom apartment with shared bathroom",
    "Generate a 10 m × 7 m two-bedroom unit. Bedroom A 4 m × 4 m + Bedroom B 3 m × 4 m + bathroom 2 m × 3 m + living 5 m × 4 m + kitchen 3 m × 3 m. Each bedroom contains bed + wardrobe; bathroom contains toilet + sink unit + bathtub; living contains sofa + coffee table; kitchen contains kitchen counter + refrigerator. Interior drywall 0.15 m; exterior concrete 0.25 m; bathroom floor ceramic tile.",
    "Wrong: placing the bathroom door so it opens directly into the kitchen violates plumbing/code separation; right: bathroom door opens onto a circulation buffer (corner of living or hallway), never directly into the kitchen.",
    [
        "expect num_rooms=5 (2 bedrooms + bathroom + living + kitchen)",
        "expect bedroom A 4 m × 4 m, bedroom B 3 m × 4 m",
        "expect bathroom 2 m × 3 m with ceramic tile floor",
        "expect bathroom door does NOT open into kitchen",
        "expect drywall 0.15 m interior, concrete 0.25 m exterior",
    ],
    ["bathroom", "ceramic", "bedroom", "kitchen", "living"]))

TASKS.append(mr_hard(29, "Studio with separate bathroom and kitchen",
    "Generate a 8 m × 6 m studio. Main room 6 m × 6 m (room_type='living', sleep+work+eat), bathroom 2 m × 3 m, kitchen 2 m × 3 m. Main room contains a bed (1.5 m × 2 m), a sofa, a desk, and a chair. Bathroom contains toilet, sink, bathtub. Kitchen contains kitchen counter, refrigerator, oven.",
    "Wrong: making the bathroom and kitchen share a wet wall with mixed plumbing risks cross-contamination; right: separate bathroom and kitchen rooms with their own walls, even if both attach to one common plumbing chase.",
    [
        "expect num_rooms=3 (main + bathroom + kitchen)",
        "expect main 6 m × 6 m, bathroom 2 m × 3 m, kitchen 2 m × 3 m",
        "expect main contains bed + sofa + desk + chair",
    ],
    ["studio", "bathroom", "kitchen", "main_room"]))

TASKS.append(mr_hard(30, "Four-room dental office",
    "Generate a 12 m × 8 m dental office suite. Reception 4 m × 4 m (reception desk + 4 waiting chairs); Exam room A 4 m × 4 m (dental chair + cabinet + light); Exam room B 4 m × 4 m (same furniture as A); Sterilization 4 m × 4 m (autoclave + counter + sink). Corridor 12 m × 2 m connects all rooms.",
    "Wrong: linking exam rooms directly to reception without a corridor breaks patient flow and HIPAA partition; right: corridor 12 m × 2 m connects all clinical rooms; reception opens to corridor only via a controlled doorway.",
    [
        "expect num_rooms=5 (reception + 2 exams + sterilization + corridor)",
        "expect each room 4 m × 4 m, corridor 12 m × 2 m",
        "expect exam rooms identical furniture set",
        "expect sterilization contains autoclave",
    ],
    ["reception", "exam", "sterilization", "corridor", "autoclave"]))

TASKS.append(mr_hard(31, "Hostel dorm cluster",
    "Generate a 14 m × 8 m hostel floor. Four 3 m × 4 m dorm rooms in a row (each with 2 bunk beds + 4 lockers) along a 14 m × 2 m corridor; one shared 2 m × 4 m bathroom and one 4 m × 2 m kitchenette at the corridor ends. Drywall interior walls 0.15 m, concrete exterior 0.25 m.",
    "Wrong: placing lockers in the corridor reduces walking width below code; right: each dorm room contains its own 4 lockers; corridor stays clear.",
    [
        "expect num_rooms=7 (4 dorms + bathroom + kitchenette + corridor)",
        "expect 4 dorms each 3 m × 4 m with 2 bunk beds + 4 lockers",
        "expect corridor 14 m × 2 m with no furniture",
        "expect shared bathroom 2 m × 4 m and kitchenette 4 m × 2 m",
    ],
    ["dorm", "bunk_bed", "locker", "corridor", "bathroom"]))

TASKS.append(mr_hard(32, "Open-plan office with breakroom and conference",
    "Generate a 14 m × 10 m office. Reception 4 m × 4 m (reception desk + waiting bench), cubicle area 10 m × 6 m (8 cubicles, each cubicle 2 m × 1.5 m fabric panels at 1.5 m height), conference 4 m × 4 m (meeting table + 8 chairs), breakroom 4 m × 4 m (round table + 4 chairs + vending machine). Carpet floor in office areas, vinyl in breakroom.",
    "Wrong: making the cubicle 'partitions' full-height walls violates open-plan intent; right: 1.5 m fabric panels are objects within the cubicle area, not room-dividing walls.",
    [
        "expect num_rooms=4 (reception + cubicle area + conference + breakroom)",
        "expect cubicle area 10 m × 6 m with 8 fabric-panel cubicles (each 2 m × 1.5 m, 1.5 m height)",
        "expect carpet floor in office areas, vinyl in breakroom",
    ],
    ["cubicle", "fabric", "conference", "breakroom"]))

# multi_room hard test
TASKS.append(mr_hard(33, "Three-bedroom with en-suite master",
    "Generate a 11 m × 9 m three-bedroom unit. Master bedroom 4 m × 5 m with en-suite bathroom 2 m × 3 m (toilet + sink + shower); bedroom B 3 m × 4 m; bedroom C 3 m × 3 m; living 4 m × 4 m; kitchen 4 m × 2 m. The en-suite bathroom is accessible only from the master bedroom.",
    "Wrong: putting a corridor door into the en-suite makes it shared, not en-suite; right: en-suite bathroom has exactly one door, on the master bedroom side.",
    [
        "expect num_rooms=6 (3 bedrooms + en-suite bathroom + living + kitchen)",
        "expect en-suite accessible only from master",
        "expect master 4 m × 5 m, bedroom B 3 m × 4 m, bedroom C 3 m × 3 m",
    ],
    ["master", "ensuite", "bathroom", "bedroom"]))

TASKS.append(mr_hard(34, "Childcare facility (play + nap + craft + restroom)",
    "Generate a 12 m × 8 m childcare facility. Play room 6 m × 6 m (play mat + toy chest + small chairs); nap room 4 m × 4 m (8 nap mats + storage shelf); craft room 4 m × 4 m (craft table + supply cabinet + 8 child chairs); child restroom 2 m × 4 m (2 child toilets + 2 sinks). Rubber floor in play, carpet in nap, vinyl in craft, ceramic in restroom.",
    "Wrong: locating the restroom adjacent only to one room blocks egress from the others; right: restroom is centrally accessible from a corridor reaching all three program rooms.",
    [
        "expect num_rooms=5 (play + nap + craft + restroom + corridor)",
        "expect rubber in play, carpet in nap, vinyl in craft, ceramic in restroom",
        "expect 8 nap mats and 8 child chairs",
    ],
    ["childcare", "play", "nap", "craft", "restroom"]))

TASKS.append(mr_hard(35, "Law office suite",
    "Generate a 12 m × 9 m law office. Lobby 4 m × 4 m (reception desk + 4 waiting chairs); 3 attorney offices each 4 m × 4 m (each: desk + bookshelf + 2 client chairs); paralegal area 8 m × 4 m (4 paralegal desks + filing cabinets). Drywall walls, carpet floor throughout, oak panel reception desk.",
    "Wrong: laying out attorney offices opening directly onto the paralegal area breaks attorney-client confidentiality; right: corridor or vestibule between each attorney office and the paralegal area.",
    [
        "expect num_rooms=6 (lobby + 3 attorney offices + paralegal + corridor)",
        "expect 3 attorney offices each 4 m × 4 m",
        "expect paralegal area 8 m × 4 m with 4 desks",
        "expect attorney offices accessed via corridor, not directly from paralegal",
    ],
    ["attorney", "paralegal", "lobby", "corridor"]))

TASKS.append(mr_hard(36, "Small clinic (waiting + 3 exams + lab + restroom)",
    "Generate a 13 m × 9 m small clinic. Waiting 4 m × 4 m (waiting bench + 6 chairs); 3 exam rooms each 3 m × 4 m (exam table + doctor stool + supply cart); lab 4 m × 4 m (workbench + lab stool + centrifuge); restroom 2 m × 4 m (toilet + sink + grab bar).",
    "Wrong: routing the corridor between exams and lab so contaminated samples cross the patient corridor violates lab safety; right: lab has a separate corridor to receive samples, isolated from the patient waiting/exam corridor.",
    [
        "expect num_rooms=7 (waiting + 3 exams + lab + restroom + patient corridor)",
        "expect 3 exam rooms each 3 m × 4 m",
        "expect lab 4 m × 4 m with workbench + lab stool + centrifuge",
        "expect lab corridor isolated from patient corridor",
    ],
    ["clinic", "waiting", "exam", "lab", "centrifuge"]))


# ════════════════════════════════════════════════════════════════════════
# scene_edit hard (4)   T0H037–T0H039 train · T0H040 test
# ════════════════════════════════════════════════════════════════════════

def edit_hard(suffix, name, prompt, distractor, source_key, assertions_extra,
              grep_extra):
    id = f"T0H{suffix:03d}"
    split = "train" if suffix <= 39 else "test"
    grep = ["scene_edit", source_key] + grep_extra
    assertions = [
        "must create scene state",
        "must be collision free",
        "must be in bounds",
        "must have rf materials",
        f"expect input scene loaded from {FLOORPLANS[source_key]}",
        "expect output scene_state.json reflects the requested structural edit",
        "expect non-edited objects preserved unchanged",
    ] + assertions_extra
    return {
        **task(id, "scene_edit", "hard", split, name, prompt, distractor,
               assertions, grep),
        "scene_path": FLOORPLANS[source_key],
    }


TASKS.append(edit_hard(37, "Merge living and kitchen (apartment)",
    "Take the pre-shipped apartment scene_state.json. Remove the partition wall between the living room and the kitchen so they become a single open-plan room. Re-merge the room polygons (union), preserve the combined-room furniture, and remove the wall object entirely. The merged room takes the larger of the two existing room_types.",
    "Wrong: setting wall.material='air' but keeping the wall object leaves a phantom obstacle for RT; right: delete the wall record entirely and union the two room polygons into one.",
    "apartment",
    [
        "expect partition wall between living and kitchen removed",
        "expect rooms unioned into single open-plan room",
        "expect non-edited furniture preserved",
    ],
    ["merge", "union", "open_plan", "partition_removed"]))

TASKS.append(edit_hard(38, "Split open area into two cubicles (office)",
    "Take the pre-shipped office scene_state.json. Add two new fabric partitions (each 1.5 m tall, 3 m long) inside the existing open area, dividing it into three sub-zones. Each new partition is a wall object with material 'fabric' and height 1.5 m. Preserve all existing furniture and re-assign each item to its sub-zone.",
    "Wrong: adding floor-to-ceiling drywall instead of 1.5 m fabric panels changes the room's classification from open-plan to enclosed offices; right: fabric panels at 1.5 m, declared as wall objects with explicit height and material fields.",
    "office",
    [
        "expect 2 new fabric partitions (each 1.5 m tall, 3 m long) added",
        "expect existing open area divided into 3 sub-zones",
        "expect existing furniture preserved and re-assigned to sub-zones",
    ],
    ["fabric", "partition_added", "sub_zone", "cubicle"]))

TASKS.append(edit_hard(39, "Convert warehouse to co-working space (3 zones)",
    "Take the pre-shipped warehouse scene_state.json. Convert it into a co-working space with three zones: a 1/3-area collaboration zone (collab table + 6 chairs), a 1/3-area focus zone (4 individual desks + 4 chairs), and a 1/3-area amenities zone (kitchenette counter + 2 standing tables). Add two new drywall partitions to demarcate the zones. Remove any existing pallets or racking. Floor stays existing material; walls drywall.",
    "Wrong: dropping new furniture without removing the existing pallets produces collisions; right: explicitly delete pallet/racking records first, then add partitions, then add new furniture.",
    "warehouse",
    [
        "expect 3 zones (collab + focus + amenities) each ~1/3 of warehouse area",
        "expect 2 new drywall partitions added",
        "expect existing pallets/racking removed",
        "expect new furniture: collab table + 6 chairs + 4 desks + 4 chairs + kitchenette + 2 standing tables",
    ],
    ["coworking", "partition_added", "collab", "focus", "amenities"]))

# scene_edit hard test
TASKS.append(edit_hard(40, "Convert apartment to open-plan studio",
    "Take the pre-shipped apartment scene_state.json. Convert it into a single open-plan studio: remove all interior partitions (living/kitchen/bedroom dividers stay only between bathroom and main space). Union the remaining room polygons into one big living zone. Preserve the bathroom and its partition; preserve all furniture except move the bed into one corner of the new open zone.",
    "Wrong: removing the bathroom wall too creates an open toilet — violation of dignity and building code; right: preserve the bathroom enclosure and only union the bedroom/living/kitchen polygons.",
    "apartment",
    [
        "expect all interior partitions removed EXCEPT bathroom enclosure",
        "expect remaining rooms (living + kitchen + bedroom) unioned into single open zone",
        "expect bathroom and its walls preserved",
        "expect bed moved into a corner of the new open zone",
    ],
    ["open_plan", "studio", "union", "bathroom_preserved", "bed_corner"]))


# Dump
out_doc = {
    "version": "1.0",
    "tier": "T0_scene_gen",
    "split_policy": "60/40 within each difficulty bucket",
    "verifier_strategy": "grep-only via _check_generic_tokens fallback",
    "count": len(TASKS),
    "tasks": TASKS,
}
OUT.write_text(json.dumps(out_doc, indent=2))
print(f"Wrote {len(TASKS)} tasks → {OUT}")
