# Scene State Schema

`scene_state.json` is the single source of truth. All subagents read/write this file.

## Mutation Rules

1. Read current state before modifying.
2. Update `meta.modified` on every write.
3. Atomic writes (temp file → rename).
4. Never delete fields — set to `null` or `[]`.
5. IDs are stable — never reuse.
6. Coordinates: origin SW, X east, Y north, theta=0 north.
7. `orientation_deg`: degrees, 0 = facing north (+Y). In furniture entries
   this is the base rotation. Exporters must add `orientation_offset` to
   get the final rotation (see `references/export-formats.md`).

## Full Schema

```json
{
  "version": "2.0",
  "meta": {
    "created": "2024-03-15T10:00:00Z",
    "modified": "2024-03-15T10:30:00Z",
    "name": "Office Floor 3",
    "description": "Open plan office with 2 conference rooms"
  },
  "scene": {
    "type": "indoor",
    "origin": "SW",
    "coordinate_system": {
      "x": "east", "y": "north", "z": "up",
      "theta_zero": "north", "units": "meters"
    },
    "bounds": { "width": 20.0, "depth": 15.0, "height": 3.0 },
    "frequency_hz": 3.5e9,
    "source_format": null
  },
  "rooms": [{
    "id": "room_main", "name": "Main Office", "type": "office",
    "polygon": [[0,0],[20,0],[20,15],[0,15]], "height": 3.0,
    "materials": { "walls": "itu_concrete", "floor": "itu_concrete", "ceiling": "itu_plasterboard" }
  }],
  "walls": [{
    "id": "wall_1", "room_id": "room_main",
    "start": [0, 0], "end": [20, 0],
    "thickness": 0.15, "material": "itu_concrete",
    "has_window": false, "has_door": false
  }],
  "furniture": [{
    "id": "furn_desk_01", "type": "desk", "catalog_id": "3df_desk_001",
    "position": [5.0, 3.0, 0.0], "orientation_deg": 0,
    "dimensions": [1.2, 0.6, 0.75], "material": "itu_wood", "visible": true
  }],
  "transmitters": [{
    "id": "tx_1", "name": "AP-1", "position": [10.0, 7.5, 2.8],
    "power_dbm": 20.0, "frequency_hz": 3.5e9,
    "antenna": { "type": "isotropic", "pattern": null, "polarization": "V", "mimo_config": null },
    "orientation_deg": 0, "visible": true
  }],
  "receivers": [{
    "id": "rx_grid_1", "type": "grid", "height": 1.5,
    "resolution": 0.5, "bounds": [[0, 0], [20, 15]], "visible": true
  }],
  "constraints": [],
  "simulation_results": [{
    "id": "sim_001", "type": "coverage",
    "artifact_id": "ca_20240315_coverage_001",
    "timestamp": "2024-03-15T10:30:00Z", "status": "completed",
    "summary": { "mean_power_dbm": -55.3, "min_power_dbm": -82.1,
                 "max_power_dbm": -35.0, "coverage_percent_above_neg70": 87.5 }
  }],
  "export_history": [],
  "code_artifacts": []
}
```

## Related

- [defaults.md](defaults.md) — default values applied to state fields
- [export-formats.md](export-formats.md) — export mapping from state to output files
