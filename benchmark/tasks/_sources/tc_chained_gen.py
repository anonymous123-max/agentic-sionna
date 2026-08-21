"""Generator for TC (T-Chained) suite — 20 scenes × 7 capabilities = 140 tasks.

Design: define 20 indoor scenes once, then run each through 7 RF capabilities.
This makes the contribution-1-to-contribution-2 chain explicit and reduces
scene-design overhead vs. independent per-task scene specification.

Output: tc_chained.json (140 tasks)

Scenes:
  S01-S10 (easy):  rectangular single-room habitable spaces 4×3 to 7×5 m
  S11-S20 (hard):  L-shape / partitioned / multi-room / mixed-material

Capabilities (applied to each scene):
  C1 single_ap_coverage       — 1 AP, Sionna RT coverage, report coverage_pct
  C2 multi_ap_optimization    — 2-3 AP placement, report min_rss_dbm
  C3 material_frequency       — compare 2 frequencies, report coverage_diff_pp
  C4 scene_edit_recompute     — edit one furniture/wall, recompute coverage
  C5 rt_to_phy                — RT for CIR, QPSK BER at one SNR
  C6 irc_coverage_joint       — verify IRC §R303 8% window + coverage
  C7 system_level_multicell   — 2-cell PF scheduling, report fairness_index

Split (scene-based, train/test never mix):
  Train: S01-S06 (6 easy) + S11-S16 (6 hard) = 12 scenes × 7 caps = 84 tasks
  Test:  S07-S10 (4 easy) + S17-S20 (4 hard) =  8 scenes × 7 caps = 56 tasks
  Total: 140 tasks

ID scheme: TC{c}_{S{nn}} where c=1..7, nn=01..20.
e.g., TC1_S03 = C1 (single_ap_coverage) on scene S03.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / "tc_chained.json"


# ════════════════════════════════════════════════════════════════════════
# Scene definitions (20)
# ════════════════════════════════════════════════════════════════════════
# Each scene: (sid, difficulty, room_desc, dims, furniture, walls, room_type)

SCENES = [
    # --- 10 easy: rectangular single-room habitable ---
    ("S01", "easy", "home office", (5, 4),
     ["desk", "office chair", "bookshelf"],          "drywall walls",        "office"),
    ("S02", "easy", "living room", (6, 5),
     ["sofa", "coffee table", "tv stand", "armchair"],"drywall walls",        "living"),
    ("S03", "easy", "bedroom",     (4, 4),
     ["double bed", "nightstand", "wardrobe", "dresser"],"drywall walls",     "bedroom"),
    ("S04", "easy", "dining room", (6, 4),
     ["round table", "chair", "buffet"],              "drywall walls",        "dining"),
    ("S05", "easy", "conference room", (5, 4),
     ["meeting table", "chair"],                       "drywall walls",        "conference"),
    ("S06", "easy", "home library", (6, 5),
     ["bookshelf", "reading chair", "side table", "floor lamp"],"drywall walls","study"),
    # Test scenes start at S07
    ("S07", "easy", "kitchen",     (5, 4),
     ["kitchen counter", "refrigerator", "oven", "dining table"],"drywall walls","kitchen"),
    ("S08", "easy", "hobby room",  (5, 4),
     ["craft table", "supply cabinet", "craft stool", "storage shelf"],"drywall walls","living"),
    ("S09", "easy", "music room",  (5, 4),
     ["upright piano", "piano bench", "music stand", "armchair"],"drywall walls","living"),
    ("S10", "easy", "multipurpose room", (7, 5),
     ["foldable table", "chair", "storage cabinet"], "drywall walls",        "living"),

    # --- 10 hard: L-shape / partitioned / multi-room / mixed materials ---
    ("S11", "hard", "L-shaped home office + study",   (8, 4),
     ["desk", "office chair", "bookshelf", "filing cabinet"], "drywall walls (L-shape: long arm 6x4, short arm 4x3)", "office"),
    ("S12", "hard", "L-shaped living + dining",       (8, 4),
     ["sofa", "tv stand", "round table", "chair"],   "drywall walls (L-shape: long arm 7x4 living, short arm 4x3 dining)", "living"),
    ("S13", "hard", "Partitioned 2-tenant office",    (10, 6),
     ["desk", "office chair", "partition wall"],      "drywall walls + 1 full-height drywall partition at x=5", "office"),
    ("S14", "hard", "Open-plan studio with half-height kitchenette partition", (8, 6),
     ["bed", "sofa", "kitchen counter", "dining table", "half-height partition"], "drywall walls + 1 half-height drywall partition (1.2 m tall) splitting living from kitchenette", "living"),
    ("S15", "hard", "3-bedroom apartment with central corridor", (12, 9),
     ["double bed", "wardrobe"],                       "drywall walls (3 bedrooms 3x3 m each + 12x2 m corridor)", "bedroom"),
    ("S16", "hard", "2-bedroom apartment with shared bathroom",  (10, 7),
     ["double bed", "wardrobe", "sofa", "coffee table", "kitchen counter"], "drywall walls (2 bedrooms + living + kitchen + bathroom)", "living"),
    # Test scenes start at S17
    ("S17", "hard", "Two-room office with central corridor", (10, 5),
     ["desk", "office chair", "bookshelf"],            "drywall walls (2 offices 4x5 + 2x5 corridor in between)", "office"),
    ("S18", "hard", "Office suite with cubicles + meeting room", (14, 9),
     ["desk", "office chair", "meeting table", "fabric cubicle partition"], "drywall walls + 4 fabric cubicle partitions (1.5 m tall) + 1 full-height meeting-room partition", "office"),
    ("S19", "hard", "Hostel dorm cluster",            (14, 8),
     ["bunk bed", "locker"],                           "drywall walls (4 dorm rooms 3x4 each + 14x2 corridor + 4x2 shared bathroom)", "bedroom"),
    ("S20", "hard", "Studio with separate bathroom",  (8, 5),
     ["double bed", "sofa", "kitchen counter"],         "drywall walls (6x5 studio + 2x5 bathroom)", "living"),

    # --- 5 additional easy test scenes (S21-S25) ---
    ("S21", "easy", "small office",     (4, 3),
     ["desk", "office chair", "filing cabinet"],     "drywall walls",        "office"),
    ("S22", "easy", "reading nook",     (5, 4),
     ["armchair", "side table"],                     "drywall walls",        "living"),
    ("S23", "easy", "home study",       (5, 4),
     ["desk", "office chair", "bookshelf"],          "drywall walls",        "study"),
    ("S24", "easy", "small bedroom",    (4, 3),
     ["single bed", "nightstand", "dresser"],        "drywall walls",        "bedroom"),
    ("S25", "easy", "art studio",       (5, 4),
     ["easel", "drafting table", "storage cabinet"], "drywall walls",        "living"),

    # --- 5 additional hard test scenes (S26-S30) ---
    ("S26", "hard", "Open-plan dining + kitchen",     (8, 5),
     ["dining table", "kitchen counter"],            "drywall walls (open-plan, no partition)", "living"),
    ("S27", "hard", "Split office: 2 zones",          (10, 6),
     ["desk", "office chair", "bookshelf"],          "drywall walls + 1 full-height drywall partition at x=5 dividing 2 work zones", "office"),
    ("S28", "hard", "Two-bedroom apartment",          (10, 7),
     ["double bed", "sofa"],                          "drywall walls (2 bedrooms 4x3 + living 4x4)", "bedroom"),
    ("S29", "hard", "Large open studio apartment",    (12, 8),
     ["double bed", "sofa"],                          "drywall walls (single open zone, no partitions)", "living"),
    ("S30", "hard", "Home office + meeting room",     (8, 5),
     ["desk", "office chair", "meeting table"],      "drywall walls + 1 full-height drywall partition (5x5 office + 3x5 meeting)", "office"),
]
assert len(SCENES) == 30


def scene_to_phrase(scene):
    sid, diff, name, dims, furn, walls, _ = scene
    furn_phrase = ", ".join(f"one {f}" for f in furn)
    return (f"a {dims[0]} m × {dims[1]} m {name} ({walls}) containing {furn_phrase}")


def scene_dims_token(scene):
    return f"{scene[3][0]}_x_{scene[3][1]}"


def scene_furniture_token(scene):
    return "_".join(f.replace(" ", "_").replace("-", "_") for f in scene[4])


def scene_train_or_test(scene):
    """S01-S20 train (20 scenes), S21-S30 test (10 scenes)."""
    sid = scene[0]
    num = int(sid[1:])
    return "train" if num <= 20 else "test"


# ════════════════════════════════════════════════════════════════════════
# Task builder factory
# ════════════════════════════════════════════════════════════════════════

def task(id, capability, difficulty, split, name, prompt, distractor,
         assertions, extra_subchecks, scene_path=None):
    core = [
        {"key": "must_create_scene_state",       "type": "file_exists"},
        {"key": "must_create_simulation_result", "type": "file_exists"},
        {"metric": "collision_free_check",        "type": "code_contains"},
        {"metric": "in_bounds_check",             "type": "code_contains"},
        {"metric": "sionna_loadable_check",       "type": "code_contains"},
    ]
    base_assertions = [
        "must create scene state",
        "must create simulation result",
        "must be collision free",
        "must be in bounds",
        "must be loadable by Sionna RT",
    ]
    return {
        "id": id,
        "origin": "redesign_chained",
        "origin_id": id,
        "tier": "TC_chained",
        "capability": capability,
        "difficulty": difficulty,
        "split": split,
        "name": name,
        "prompt": prompt,
        "distractor": distractor,
        "scene_path": scene_path,
        "required_artifacts": ["scene_state.json", "simulation_result.json"],
        "assertions": base_assertions + assertions,
        "verifier": {
            "type": "composite",
            "subchecks": core + extra_subchecks,
        }
    }


# ════════════════════════════════════════════════════════════════════════
# Capability task builders — each takes a scene, returns a task
# ════════════════════════════════════════════════════════════════════════

def make_c1(scene):
    """C1: single AP coverage."""
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    freq = 5.0 if diff == "easy" else 3.5
    thr = -75
    task_id = f"TC1_{sid}"
    task_name = f"C1 single AP coverage on {sid} ({name})"
    prompt = (f"Generate {scene_phrase}. Then place one AP at the centroid at 2.5 m height. "
              f"Compute coverage at {freq} GHz using Sionna RT (RadioMapSolver), "
              f"threshold {thr} dBm. Report coverage_pct in simulation_result.json.\n\n"
              f"⚠ Use Sionna RT — do NOT fall back to analytical FSPL. The point is to "
              f"test that you can drive the ray-tracer on the scene you generated. "
              f"If `import sionna` fails, fix the environment (use the conda env "
              f"specified by $RF_SIONNA_PY) and retry — do not silently switch to "
              f"a hand-coded path-loss formula.")
    distractor = (f"Wrong: skipping scene generation and using a default 10×10 room. "
                  f"Right: generate the {name} first via the scene_gen pipeline, "
                  f"then place TX at the centroid and run Sionna RT.")
    assertions = [
        f"expect room dims {dims[0]} m × {dims[1]} m",
        f"expect single AP at {freq} GHz",
        f"expect expected_furniture={furn}",
        f"expect coverage_pct >= {50 if diff == 'easy' else 30}",
    ]
    extra = [
        {"metric": "coverage_pct", "type": "metric_range",
         "min": (50 if diff == "easy" else 30), "max": 100.0},
        {"metric": scene_furniture_token(scene), "type": "code_contains"},
        {"metric": "rt_oracle_check", "type": "code_contains"},
        # Reference oracle: analytical FSPL grid as numerical ground truth
        {"metric": "c1_ref_oracle_check", "type": "code_contains"},
        # FSPL-fallback guard: simulation.py must `import sionna` and
        # simulation_result.json must not flag an analytical fallback.
        {"metric": "sionna_rt_used_check", "type": "code_contains"},
    ]
    return task(task_id, "single_ap_coverage", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


def make_c2(scene):
    """C2: multi-AP optimization."""
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    n_aps = 2 if diff == "easy" else 3
    task_id = f"TC2_{sid}"
    task_name = f"C2 {n_aps}-AP optimization on {sid}"
    prompt = (f"Generate {scene_phrase}. Then place {n_aps} APs at 5 GHz to maximize the "
              f"minimum-RSS across the floor (threshold -75 dBm). Report ap_positions, "
              f"min_rss_dbm, and coverage_pct in simulation_result.json.")
    distractor = (f"Wrong: placing all {n_aps} APs on the same wall wastes range. "
                  f"Right: spread APs along the long axis for spatial diversity.")
    assertions = [
        f"expect {n_aps} APs placed",
        f"expect ap_positions[] has {n_aps} entries",
        f"expect min_rss_dbm >= -85",
    ]
    extra = [
        {"metric": "min_rss_dbm", "type": "metric_threshold",
         "threshold": -85, "direction": ">="},
        {"metric": "two_aps" if n_aps == 2 else "three_aps", "type": "code_contains"},
        {"metric": "rt_oracle_check",       "type": "code_contains"},
        {"metric": "geometry_oracle_check", "type": "code_contains"},
    ]
    return task(task_id, "multi_ap_optimization", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


def make_c3(scene):
    """C3: material/frequency comparison."""
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    freqs = (2.4, 5.0) if diff == "easy" else (5.0, 28.0)
    f1, f2 = freqs
    task_id = f"TC3_{sid}"
    task_name = f"C3 {f1}/{f2} GHz comparison on {sid}"
    prompt = (f"Generate {scene_phrase}. Place one AP at the centroid. "
              f"Compute coverage at BOTH {f1} GHz and {f2} GHz. Report under "
              f"simulation_result.json: numerical_metrics.coverage_pct_low_freq, "
              f"numerical_metrics.coverage_pct_high_freq, numerical_metrics.coverage_diff_pp = "
              f"coverage(low) − coverage(high). ALSO emit simulation_config.frequencies_ghz = "
              f"[{f1}, {f2}] so the reference oracle can re-derive both values analytically. "
              f"(Legacy keys coverage_pct_{int(f1*10)}_ghz / _{int(f2*10)}_ghz also accepted.)")
    distractor = (f"Wrong: simulating only one frequency. Right: run the same TX through "
                  f"both bands; expect higher freq to give lower coverage due to FSPL.")
    assertions = [
        f"expect both frequencies tested: {f1} and {f2} GHz",
        "expect coverage_diff_pp reported",
    ]
    extra = [
        {"metric": "coverage_diff_pp", "type": "metric_range", "min": -50, "max": 100},
        {"metric": f"{int(f1*10)}_ghz", "type": "code_contains"},
        {"metric": f"{int(f2*10)}_ghz", "type": "code_contains"},
        {"metric": "rt_oracle_check", "type": "code_contains"},
        # Reference oracle: analytical FSPL at both freqs + diff arithmetic
        {"metric": "c3_ref_oracle_check", "type": "code_contains"},
    ]
    return task(task_id, "material_frequency", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


def make_c4(scene):
    """C4 / T2: scene_edit + recompute coverage.

    Design v3 (2026-05-19): CONCRETE partition + threshold -50 dBm.
      - Edit: add a full-height *concrete* interior wall at x = W/2 (ALL scenes)
        with `material: itu_concrete`. Concrete attenuates 17-20 dB at 5 GHz
        (drywall only 4-5 dB), so far-side coverage clearly drops.
      - Threshold: -50 dBm — strict enough that far side flips below it.
      - Frequency: 5 GHz (keep — consistent with T1)
      - REQUIRES Sionna RT (FSPL fallback ignores walls → 0 delta).

    Rationale: v2 used drywall + -65 dBm and 16/20 trials showed before=after
    because (a) drywall is too weak and (b) agent often fell back to FSPL
    which doesn't model walls at all. Concrete + strict threshold + RT-only
    guidance forces a real RT computation with visible delta.
    """
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    edit_desc = (f"add a full-height interior wall at x={dims[0]/2:.1f} m "
                 f"splitting the room into two halves, with material `itu_concrete`")
    action = "added"
    threshold_dbm = -50  # strict; drywall partition would not budge coverage
    task_id = f"TC4_{sid}"
    task_name = f"T2 scene_edit (concrete partition) on {sid}"
    prompt = (f"Generate {scene_phrase}. Place one AP at the centroid at 2.5 m height, "
              f"frequency 5.0 GHz, power 20 dBm. Coverage threshold {threshold_dbm} dBm.\n\n"
              f"Step 1 — Compute baseline coverage of the room as generated. Save the "
              f"resulting RSS grid as coverage_map_before.npy and report "
              f"numerical_metrics.coverage_pct_before in simulation_result.json.\n\n"
              f"Step 2 — Edit the scene: {edit_desc}. Recompute coverage with the SAME "
              f"AP and the concrete partition now in place. Save the new RSS grid as "
              f"coverage_map_after.npy and report numerical_metrics.coverage_pct_after AND "
              f"numerical_metrics.coverage_delta_pp = coverage_pct_after − coverage_pct_before "
              f"(signed).\n\n"
              f"⚠ CRITICAL: Use Sionna RT (RadioMapSolver). FSPL fallback "
              f"DOES NOT model walls, so a partition would have ZERO effect under "
              f"FSPL and your delta would be 0 → AUTOMATIC FAIL. The point of "
              f"this task is to test that you can model wall attenuation. Even if "
              f"sionna throws errors on the first try, retry / fix the scene XML / "
              f"shrink the grid; do NOT fall back to FSPL.\n\n"
              f"The reference oracle checks: (i) both before/after present, "
              f"(ii) delta = after − before within ±2 pp, (iii) sign matches edit "
              f"(action=added → expect Δ ≤ +5 pp; concrete partition typically "
              f"gives Δ between -30 and -55 pp).")
    distractor = (f"Wrong: not recomputing after the edit just reports baseline twice. "
                  f"Right: write two simulations (before and after the {action} step) "
                  f"and report a non-zero delta.")
    assertions = [
        f"expect edit applied: {edit_desc}",
        "expect coverage_pct_before and coverage_pct_after both reported",
        "expect non-zero coverage_delta_pp",
    ]
    extra = [
        {"metric": "coverage_delta_pp", "type": "metric_range",
         "min": -100, "max": 100},
        {"metric": "coverage_pct_before", "type": "code_contains"},
        {"metric": "coverage_pct_after",  "type": "code_contains"},
        # NOTE: dropped the `code_contains: added` action-verb grep — it's
        # redundant with c4_ref_oracle (which already enforces sign+arithmetic)
        # and was failing on agents that wrote "add" / "adding" / "partition"
        # but not the past-tense "added" literal.
        {"metric": "rt_oracle_check",       "type": "code_contains"},
        {"metric": "geometry_oracle_check", "type": "code_contains"},
        # Reference oracle: arithmetic consistency + sign by edit type
        {"metric": "c4_ref_oracle_check", "type": "code_contains",
         "edit_action": action},
        # FSPL-fallback guard
        {"metric": "sionna_rt_used_check", "type": "code_contains"},
    ]
    return task(task_id, "scene_edit_recompute", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


def make_c5(scene):
    """C5 / T3: RT-to-PHY chain (CIR + QPSK BER).

    Design notes (May 2026):
      - Fixed TX and RX positions (corner-to-corner) so all 20 scenes
        share a comparable link geometry.
      - Threshold band uses analytical AWGN floor + 1.5×Rayleigh ceiling
        (see _check_c5_reference_oracle).
      - SNR 10 dB easy / 12 dB hard.
    """
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    snr_db = 10 if diff == "easy" else 12
    mod = "QPSK"
    freq_ghz = 2.4
    rx_x = max(dims[0] - 1, 2)
    rx_y = max(dims[1] - 1, 2)
    task_id = f"TC5_{sid}"
    task_name = f"T3 RT→PHY {mod} BER on {sid}"
    prompt = (
        f"Generate {scene_phrase}. Place 1 TX at (1, 1, 2.5) m and 1 RX at "
        f"({rx_x}, {rx_y}, 1.5) m. Operating frequency {freq_ghz} GHz "
        f"(WiFi-2.4 band), modulation {mod}, SNR = {snr_db} dB.\n\n"
        f"Step 1 — Compute the channel impulse response (CIR) for the TX↔RX "
        f"link via Sionna RT's PathSolver / Paths API. Do NOT fall back to "
        f"analytical FSPL — the point is to test that you can extract a real "
        f"multipath CIR. Use the sionna conda env if `import sionna` fails "
        f"(see $RF_SIONNA_PY). Save:\n"
        f"  numerical_metrics.cir_path_count        : int, number of multipath components\n"
        f"  numerical_metrics.path_delays_ns        : list[float], multipath delays (ns)\n"
        f"  numerical_metrics.path_gains_db         : list[float], multipath gains (dB)\n\n"
        f"Step 2 — Simulate uncoded {mod} BER over that CIR at SNR={snr_db} dB. "
        f"Convolve symbols with the CIR (the multipath spreads symbols → ISI). "
        f"Run ≥ 10000 symbols for statistical reliability.\n\n"
        f"⚠ BER definition: count BIT errors, not symbol errors. For {mod} use "
        f"Gray-coded 2 bits/symbol, then BER = bit_errors / total_bits. "
        f"Uncoded BER for any binary modulation is physically bounded by "
        f"BER ∈ [0, 0.5]; anything > 0.5 means you reported SER instead of BER.\n\n"
        f"Save:\n"
        f"  numerical_metrics.ber                   : float ∈ [0, 0.5], simulated multipath BER\n"
        f"  numerical_metrics.ber_theoretical_awgn  : float, ideal AWGN baseline at "
        f"the same SNR (use Q(√(2·γ)) = 0.5·erfc(√γ))\n"
        f"  numerical_metrics.snr_db                : {snr_db}\n\n"
        f"The reference oracle checks: (i) ber ∈ [AWGN floor, max(1.5·Rayleigh, 0.5)] "
        f"— allows ISI-dominated regime but rejects > 0.5; (ii) ber_theoretical_awgn "
        f"matches the analytical Q-function within 3×; (iii) ber > AWGN-floor (the "
        f"multipath must DEGRADE not improve the link vs AWGN). AWGN-only fallbacks "
        f"(ber = ber_theoretical_awgn exactly) will FAIL."
    )
    distractor = (f"Wrong: reporting AWGN BER (ignoring multipath) at the same SNR. "
                  f"Right: convolve the symbols with the simulated CIR; expect ber to differ "
                  f"from ber_theoretical_awgn at SNR={snr_db} dB.")
    assertions = [
        "expect 1 TX + 1 RX pair specified",
        "expect cir_path_count >= 1",
        f"expect ber and ber_theoretical_awgn both reported at SNR={snr_db} dB",
    ]
    extra = [
        {"metric": "ber", "type": "metric_range", "min": 0.0, "max": 1.0},
        {"metric": "cir", "type": "code_contains"},
        {"metric": "qpsk", "type": "code_contains"},
        {"metric": "phy_oracle_check", "type": "code_contains"},
        # Reference oracle: AWGN/Rayleigh bounds + AWGN baseline sanity
        {"metric": "c5_ref_oracle_check", "type": "code_contains"},
        # FSPL-fallback guard
        {"metric": "sionna_rt_used_check", "type": "code_contains"},
    ]
    return task(task_id, "rt_to_phy", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


def make_c6(scene):
    """C6: IRC §R303 + coverage joint."""
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    floor_area = dims[0] * dims[1]
    min_aperture = floor_area * 0.08
    freq = 28.0 if diff == "easy" else 5.0  # easy: mmWave, hard: 5G NR
    task_id = f"TC6_{sid}"
    task_name = f"C6 IRC + {freq} GHz coverage on {sid}"
    prompt = (f"Generate {scene_phrase} (room_type='{rtype}'). Add one or more windows on perimeter "
              f"walls totaling at least {min_aperture:.2f} m² (8% of {floor_area:.1f} m² floor area, "
              f"per IRC §R303). Place one AP at the centroid. Compute coverage at {freq} GHz. "
              f"Report coverage_pct, irc_compliant (true/false), total_window_aperture_m2.")
    distractor = (f"Wrong: putting a 'window' on an interior wall (not facing exterior) — "
                  f"IRC requires perimeter wall windows. Right: place windows on north/south/east/west "
                  f"perimeter walls only.")
    assertions = [
        f"expect room of type '{rtype}'",
        f"expect window aperture >= {min_aperture:.2f} m²",
        "expect all windows on perimeter walls",
        "expect coverage_pct AND irc_compliant both reported",
    ]
    extra = [
        {"metric": "coverage_pct", "type": "metric_range", "min": 30, "max": 100},
        {"metric": "irc_compliant", "type": "code_contains"},
        {"metric": "perimeter", "type": "code_contains"},
        {"metric": "aperture", "type": "code_contains"},
        {"metric": "rt_oracle_check", "type": "code_contains"},
    ]
    return task(task_id, "irc_coverage_joint", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


def make_c7(scene):
    """C7: 2-cell multi-cell scheduling."""
    sid, diff, name, dims, furn, walls, rtype = scene
    scene_phrase = scene_to_phrase(scene)
    n_cells = 2 if diff == "easy" else 3
    n_users = max(4, len(furn))
    task_id = f"TC7_{sid}"
    task_name = f"C7 {n_cells}-cell PF scheduling on {sid}"
    prompt = (f"Generate {scene_phrase}. Deploy {n_cells} cells (APs as base stations) "
              f"at evenly-spaced positions, 3.5 GHz. Place {n_users} users uniformly. "
              f"Run proportional-fair (PF) scheduling for 100 TTI. "
              f"Report under simulation_result.json.numerical_metrics: "
              f"per_user_avg_rate (length-{n_users} list of non-negative floats, bps/Hz), "
              f"mean_throughput_bps_hz = mean(per_user_avg_rate), "
              f"fairness_index = Jain's = (Σr_i)² / (n · Σr_i²). "
              f"The reference oracle re-derives Jain's from per_user_avg_rate[] and rejects "
              f"results where reported fairness or mean throughput don't match (±0.05 / ±0.2 bps/Hz).")
    distractor = (f"Wrong: simulating only one cell and treating each user separately. "
                  f"Right: run a multi-cell scheduler that distributes capacity across all {n_users} "
                  f"users with PF metric T^(1−α)/r.")
    assertions = [
        f"expect {n_cells} cells deployed",
        f"expect {n_users} users",
        "expect PF scheduling for 100 TTI",
        "expect fairness_index >= 0.6",
    ]
    extra = [
        {"metric": "fairness_index", "type": "metric_threshold",
         "threshold": 0.6, "direction": ">="},
        {"metric": "two_cells" if n_cells == 2 else "three_cells", "type": "code_contains"},
        {"metric": "pf_scheduling", "type": "code_contains"},
        {"metric": "sys_oracle_check", "type": "code_contains"},
        # Reference oracle: re-compute Jain + mean_thr from per_user_avg_rate[]
        {"metric": "c7_ref_oracle_check", "type": "code_contains"},
    ]
    return task(task_id, "system_level_multicell", diff, scene_train_or_test(scene),
                task_name, prompt, distractor, assertions, extra)


# ════════════════════════════════════════════════════════════════════════
# Generate all (scene × capability) cross-product tasks
# ════════════════════════════════════════════════════════════════════════

CAPABILITY_BUILDERS = [
    ("C1", make_c1),
    ("C2", make_c2),
    ("C3", make_c3),
    ("C4", make_c4),
    ("C5", make_c5),
    ("C6", make_c6),
    ("C7", make_c7),
]

TASKS = []
for scene in SCENES:
    for cap_label, builder in CAPABILITY_BUILDERS:
        TASKS.append(builder(scene))

# ════════════════════════════════════════════════════════════════════════
# Dump
# ════════════════════════════════════════════════════════════════════════
from collections import Counter

out_doc = {
    "version": "3.0",
    "tier": "TC_chained",
    "design": "20 scenes × 7 capabilities = 140 chained tasks",
    "split_policy": "scene-based: 12 train scenes × 7 cap = 84 train tasks; "
                    "8 test scenes × 7 cap = 56 test tasks",
    "verifier_strategy": "composite (file_exists + collision + in_bounds + metric_range/threshold + token grep)",
    "count": len(TASKS),
    "tasks": TASKS,
}
OUT.write_text(json.dumps(out_doc, indent=2))
print(f"Wrote {len(TASKS)} chained tasks -> {OUT}")
print(f"\nSplit:      {dict(Counter(t['split'] for t in TASKS))}")
print(f"Difficulty: {dict(Counter(t['difficulty'] for t in TASKS))}")
print(f"\nBy capability:")
for cap in sorted({t['capability'] for t in TASKS}):
    n_tr = sum(1 for t in TASKS if t['capability']==cap and t['split']=='train')
    n_te = sum(1 for t in TASKS if t['capability']==cap and t['split']=='test')
    n_e = sum(1 for t in TASKS if t['capability']==cap and t['difficulty']=='easy')
    n_h = sum(1 for t in TASKS if t['capability']==cap and t['difficulty']=='hard')
    print(f"  {cap:<28}  train={n_tr:>2}  test={n_te:>2}  (easy={n_e:>2} + hard={n_h:>2})")
print(f"\nBy scene (sample): TC1_S01..TC7_S20 are all 7×20 = 140 combinations")
