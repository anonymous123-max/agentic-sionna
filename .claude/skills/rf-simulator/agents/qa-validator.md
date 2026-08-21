# QA Validation Checklist

Read this after running a script and before showing results to the user.
When you encounter and fix a new error, add it to `references/error-patterns.md`
so it's caught automatically next time.

---

## Checks by Output Type

### After scene creation

| When this is true | Check this | It fails because |
|---|---|---|
| scene_state.json was written | JSON parses without error | Crash during write can corrupt the file |
| Scene has objects | Required fields present: version, meta, scene, rooms, transmitters | Missing fields break downstream templates |
| Scene has multiple objects | All `id` fields unique | Duplicate IDs cause Sionna to silently merge objects |
| Furniture was placed | No bounding-box overlaps | Overlapping furniture produces unrealistic layouts |
| TX/RX was placed | Positions within `scene.bounds` | Out-of-bounds TX produces all-zero coverage |

### After simulation

| When this is true | Check this | It fails because |
|---|---|---|
| Script ran | Exit code 0 and coverage_map.npy exists | Silent crashes produce empty output dirs |
| Coverage data exists | ≥10% non-zero cells | All-zero means TX outside room or frequency not set |
| Coverage data exists | RSS values in [-120, +30] dBm range | Values outside this range violate energy conservation |
| Coverage data exists | RSS never exceeds TX power + 0.1 dB | Higher RSS means free energy — physics bug |
| Script wrote CodeArtifact | Hash matches script content | Mismatch means script was edited after artifact creation |

### After export

| When this is true | Check this | It fails because |
|---|---|---|
| XML was generated | `xml.etree.ElementTree` parses it | Malformed XML crashes Sionna scene loading |
| GLB was generated | File has valid GLTF header | Corrupt GLB shows blank viewer |
| Any export ran | `orientation_offset` applied to rotation | Missing offset rotates furniture to wrong orientation |

See `references/physics-validation.md` for detailed formulas and thresholds.

---

## Error Pattern Matching

When a script fails:
1. Search `references/error-patterns.md` for matching error type + message
2. If match found → apply documented fix → re-run
3. If no match and you fix it → add the new pattern so it's caught next time

Format for new entries:
```
### EP-<number>: <description>
**Error type**: `ExceptionType`  **Message**: `substring`
**Root cause**: why  **Fix**: steps  **Added**: date
```

Only add patterns for errors that were successfully fixed and are
reproducible — untested patterns pollute the knowledge base.

---

## Scoring

| Criterion | Points | Why it matters |
|---|---|---|
| Script completes (exit 0) | 15 | Crashes produce no output at all |
| Coverage physically plausible | 10 | Wrong physics invalidates all downstream analysis |
| Coverage map non-zero | 10 | Zero map means fundamental configuration error |
| CodeArtifact created + hash match | 10 | Provenance tracking for reproducibility |
| Physics ERROR checks pass | 10 | Energy conservation violations |
| Physics WARNING checks pass | 10 | Monotonicity / continuity issues |
| Export files valid | 15 | Viewer won't load corrupt files |
| Runtime < 30s | 10 | Timeouts kill the agent loop |
| Constraints evaluated | 10 | User-specified requirements verified |

| Score | Action |
|---|---|
| 90+ | Deliver to user |
| 70-89 | Deliver with warnings noted |
| 50-69 | Fix before delivering |
| <50 | Do not deliver — escalate |
