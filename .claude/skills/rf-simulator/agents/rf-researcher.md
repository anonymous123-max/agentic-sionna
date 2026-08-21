# RF Researcher Reference

Procedures for monitoring Sionna releases, researching API changes, and
keeping reference files current. Read-only — never modify code or run sims.

---

## Trigger Conditions

| Trigger | When | Action |
|---|---|---|
| T1: New PyPI release | `pip index versions sionna` shows newer version | Full update flow |
| T2: Unknown API | ImportError/AttributeError for sionna.* symbol | Search docs, update sionna-v2-api.md |
| T3: User question | User asks about undocumented Sionna feature | Research and add to reference files |
| T4: Session start | First simulation request | Version check only |
| T5: Explicit request | User says "check for updates" or "research X" | Targeted research |
| T6: Discovery request | Skill-improver needs research on uncovered topic | Search docs/arXiv, return structured findings |
| T7: Paper-to-tool | T6 finds paper with public code repo | Assess repo, return integration notes |

---

## Update Flow

1. **Version discovery**: `pip show sionna` vs `pip index versions sionna`
2. **Fetch changelog**: GitHub releases → PyPI → CHANGELOG.md
3. **Diff against references**: Compare changelog to `references/sionna-v2-api.md`
4. **Update files**: New/changed APIs → sionna-v2-api.md. Breaking changes → error-patterns.md. New patterns → script-guidelines.md.

---

## Research Topics

- **RT1**: RF-3DGS (3D Gaussian Splatting for RF, neural radiance fields)
- **RT2**: Propagation model updates (ITU-R P.1238, 3GPP TR 38.901)
- **RT3**: Sionna integration patterns (multi-GPU, memory, performance)
- **RT4**: Competing tools (WinProp, Ranplan, iBwave, CloudRF)

---

## Constraints

**Does**: Read external sources, update `references/` files, report findings.
**Does NOT**: Install packages, modify scripts, modify scene_state.json, run simulations.
