# Data Source Acquisition Guides

## Contents

1. [Meta Aria Synthetic Environments (ASE)](#meta-aria-synthetic-environments-ase) — High-fidelity indoor 3D scenes with furniture and materials
2. [OpenStreetMap (OSM)](#openstreetmap-osm) — Building footprints and heights for outdoor urban scenes
3. [Furniture Catalogs](#furniture-catalogs) — 3D-FUTURE and other OBJ mesh sources for indoor objects
4. [Data Source Selection Guide](#data-source-selection-guide) — Decision matrix for choosing the right data source

Reference for obtaining 3D scene data, building geometry, and furniture models for RF simulation.

---

## Meta Aria Synthetic Environments (ASE)

### What It Is

Aria Synthetic Environments is a dataset of 100,000 procedurally-generated indoor scenes created by Meta for training embodied AI agents. Each scene includes:

- Full 3D mesh geometry (walls, floors, ceilings, doors, windows)
- Placed furniture with category labels
- Multi-view RGB, depth, and instance segmentation renders
- Scene description in ASE Scene Language (a text-based DSL)
- Simulated sensor trajectories (for Aria glasses research)

The scenes span residential layouts (apartments, houses) with realistic room connectivity, furniture placement, and material assignments.

### Access and License

1. Visit [projectaria.com](https://www.projectaria.com/datasets/ase/)
2. Sign the license agreement (research and non-commercial use)
3. Receive download credentials

### Download via projectaria_tools

```bash
# Install the tools package
pip install projectaria-tools

# Download the CDN manifest (JSON file listing all scenes)
python -m projectaria_tools.ase.download_manifest \
    --output ase_manifest.json

# Download a subset of scenes (e.g., first 100)
python -m projectaria_tools.ase.download_scenes \
    --manifest ase_manifest.json \
    --output-dir ./ase_data \
    --start 0 --count 100
```

### Data Format (Per Scene Directory)

```
scene_XXXXX/
  ase_scene_language.txt    # Text description of scene geometry
  mesh/
    scene.glb               # Full 3D mesh (GLTF binary)
  rgb/
    frame_XXXXX.jpg         # Rendered RGB images
  depth/
    frame_XXXXX.png         # Depth maps (16-bit PNG, millimeters)
  instances/
    frame_XXXXX.png         # Instance segmentation masks
  trajectory.csv            # Camera poses per frame
  scene_metadata.json       # Room dimensions, furniture list, materials
```

### ASE Scene Language

The `ase_scene_language.txt` file contains procedural commands that fully describe the scene geometry:

```
# Example ASE Scene Language commands
make_wall start=(0,0) end=(5,0) height=2.8 thickness=0.15 material=concrete
make_wall start=(5,0) end=(5,4) height=2.8 thickness=0.15 material=concrete
make_wall start=(5,4) end=(0,4) height=2.8 thickness=0.15 material=concrete
make_wall start=(0,4) end=(0,0) height=2.8 thickness=0.15 material=concrete
make_door wall=2 position=2.0 width=0.9 height=2.1
make_window wall=1 position=2.5 width=1.2 height=1.4 sill_height=0.9
place_furniture type=sofa position=(2.5, 0.5) rotation=0 model=sofa_0042
place_furniture type=table position=(2.5, 2.0) rotation=90 model=table_0117
```

### Conversion Pipeline to Sionna

Parse ASE scenes into the project's Scene model, then export to Mitsuba XML for Sionna:

```python
import re
from pathlib import Path

def parse_ase_scene(ase_text: str) -> dict:
    """Parse ASE Scene Language into structured geometry data.

    Returns dict with 'walls', 'doors', 'windows', 'furniture' lists.
    """
    result = {"walls": [], "doors": [], "windows": [], "furniture": []}

    for line in ase_text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("make_wall"):
            params = _parse_params(line)
            result["walls"].append({
                "start": _parse_tuple(params["start"]),
                "end": _parse_tuple(params["end"]),
                "height": float(params.get("height", 2.8)),
                "thickness": float(params.get("thickness", 0.15)),
                "material": params.get("material", "concrete"),
            })

        elif line.startswith("make_door"):
            params = _parse_params(line)
            result["doors"].append({
                "wall_index": int(params["wall"]),
                "position": float(params["position"]),
                "width": float(params.get("width", 0.9)),
                "height": float(params.get("height", 2.1)),
            })

        elif line.startswith("make_window"):
            params = _parse_params(line)
            result["windows"].append({
                "wall_index": int(params["wall"]),
                "position": float(params["position"]),
                "width": float(params.get("width", 1.2)),
                "height": float(params.get("height", 1.4)),
                "sill_height": float(params.get("sill_height", 0.9)),
            })

        elif line.startswith("place_furniture"):
            params = _parse_params(line)
            result["furniture"].append({
                "type": params["type"],
                "position": _parse_tuple(params["position"]),
                "rotation": float(params.get("rotation", 0)),
                "model_id": params.get("model", None),
            })

    return result


def _parse_params(line: str) -> dict:
    """Extract key=value pairs from an ASE command line."""
    params = {}
    for match in re.finditer(r'(\w+)=([^\s]+)', line):
        params[match.group(1)] = match.group(2)
    return params


def _parse_tuple(s: str) -> tuple:
    """Parse '(x,y)' or '(x,y,z)' string to float tuple."""
    s = s.strip("()")
    return tuple(float(v) for v in s.split(","))
```

**Material classification for RF simulation:**

| ASE Material   | ITU-R P.2040 Material | Sionna Material Name |
|----------------|----------------------|----------------------|
| concrete       | Concrete             | itu_concrete         |
| drywall        | Plasterboard         | itu_plasterboard     |
| brick          | Brick                | itu_brick            |
| glass          | Glass                | itu_glass            |
| wood           | Wood                 | itu_wood             |
| metal          | Metal                | itu_metal            |
| plaster        | Plasterboard         | itu_plasterboard     |

---

## OpenStreetMap (OSM)

### Three Query Modes

#### 1. Lat/Lon + Radius

```python
import osmnx as ox

# Download buildings within 500m of a point
buildings = ox.features_from_point(
    center_point=(37.7749, -122.4194),  # (lat, lon) - San Francisco
    dist=500,                            # radius in meters
    tags={"building": True},
)
```

#### 2. Bounding Box

```python
# Download buildings in a bounding box
# Order: (north, south, east, west) -- NOTE: osmnx convention
buildings = ox.features_from_bbox(
    bbox=(37.78, 37.77, -122.41, -122.42),
    tags={"building": True},
)
```

#### 3. Location Name (Geocoded)

```python
# Download by place name (uses Nominatim geocoder)
buildings = ox.features_from_place(
    query="Quartier Latin, Paris, France",
    tags={"building": True},
)
```

### osmnx Library Usage

```bash
pip install osmnx
```

**Important:** osmnx requires network access to query the Overpass API. For offline use or CI environments, gate with:

```python
try:
    import osmnx as ox
except ImportError:
    raise ImportError(
        "osmnx is required for OSM data. Install with: pip install osmnx"
    )
```

The project convention uses lazy imports with helpful ImportError messages (see MEMORY.md).

### Building Height Heuristics

OSM buildings rarely have explicit height tags. Apply these defaults based on the `building` tag value:

```python
DEFAULT_BUILDING_HEIGHTS = {
    # Residential
    "residential": 8.0,
    "apartments": 15.0,
    "house": 7.0,
    "detached": 7.0,
    "terrace": 8.0,

    # Commercial
    "commercial": 15.0,
    "office": 20.0,
    "retail": 6.0,
    "supermarket": 6.0,

    # Industrial
    "industrial": 12.0,
    "warehouse": 10.0,
    "garage": 4.0,

    # Public
    "school": 12.0,
    "university": 15.0,
    "hospital": 20.0,
    "church": 15.0,

    # Generic
    "yes": 10.0,       # unspecified building type
    "construction": 10.0,
}

def get_building_height(building_tags: dict) -> float:
    """Extract or estimate building height in meters.

    Priority: explicit height tag > building:levels * 3m > heuristic by type.
    """
    # 1. Explicit height tag
    if "height" in building_tags:
        try:
            return float(str(building_tags["height"]).replace("m", "").strip())
        except ValueError:
            pass

    # 2. Levels-based estimate (3 meters per level)
    if "building:levels" in building_tags:
        try:
            return float(building_tags["building:levels"]) * 3.0
        except ValueError:
            pass

    # 3. Heuristic by building type
    building_type = str(building_tags.get("building", "yes")).lower()
    return DEFAULT_BUILDING_HEIGHTS.get(building_type, 10.0)
```

### Coordinate Transforms

OSM data uses WGS84 (lat/lon). Sionna scenes use local Cartesian meters. Convert via UTM projection:

```python
import numpy as np

def latlon_to_local_meters(
    lats: np.ndarray,
    lons: np.ndarray,
    origin_lat: float,
    origin_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert lat/lon arrays to local (x, y) in meters relative to an origin.

    Uses UTM-like equirectangular projection. Accurate for areas < 10 km.
    For larger areas, use pyproj with proper UTM zone.

    Args:
        lats: latitude values
        lons: longitude values
        origin_lat: latitude of the coordinate origin
        origin_lon: longitude of the coordinate origin

    Returns:
        (x_meters, y_meters) -- x is East, y is North
    """
    lat_rad = np.radians(origin_lat)
    m_per_deg_lat = 111_132.92
    m_per_deg_lon = 111_132.92 * np.cos(lat_rad)

    x = (lons - origin_lon) * m_per_deg_lon
    y = (lats - origin_lat) * m_per_deg_lat

    return x, y


def polygon_latlon_to_meters(polygon, origin_lat: float, origin_lon: float):
    """Convert a Shapely Polygon from lat/lon to local meters.

    Args:
        polygon: shapely.geometry.Polygon with lat/lon coordinates
        origin_lat, origin_lon: coordinate system origin

    Returns:
        List of (x, y) tuples in meters
    """
    coords = list(polygon.exterior.coords)
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])
    x, y = latlon_to_local_meters(lats, lons, origin_lat, origin_lon)
    return list(zip(x, y))
```

For higher accuracy or larger areas, use pyproj:

```python
from pyproj import Transformer

def make_utm_transformer(lat: float, lon: float):
    """Create a WGS84 -> UTM transformer for the correct UTM zone."""
    import math
    zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg = 32600 + zone if hemisphere == "north" else 32700 + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
```

### Roads and Trees (Optional)

Roads and vegetation improve RF simulation realism (signal scattering, foliage loss):

```python
# Download road network
graph = ox.graph_from_point((lat, lon), dist=500, network_type="drive")
edges = ox.graph_to_gdfs(graph, nodes=False)

# Download trees / vegetation
trees = ox.features_from_point(
    center_point=(lat, lon),
    dist=500,
    tags={"natural": ["tree", "tree_row", "wood"]},
)
```

Foliage loss models (ITU-R P.833):
- Approximate additional loss through vegetation: 0.2-1.0 dB/m at sub-6 GHz
- At mmWave: 2-5 dB/m through dense foliage

---

## Furniture Catalogs

### Amazon Berkeley Objects (ABO)

**Overview:**
- 7,953 3D product models in GLB (GLTF binary) format
- Categories: chairs, tables, beds, shelves, lamps, electronics, etc.
- Permissive license (CC BY 4.0 and Amazon License)
- Hosted on Amazon S3

**Download:**

```bash
# Install AWS CLI (no credentials needed -- public bucket)
pip install awscli

# List available model archives
aws s3 ls s3://amazon-berkeley-objects/3dmodels/ --no-sign-request

# Download all GLB models (~15 GB)
aws s3 sync s3://amazon-berkeley-objects/3dmodels/glb/ ./abo_models/ --no-sign-request

# Download metadata (category labels, dimensions)
aws s3 cp s3://amazon-berkeley-objects/listings/metadata/ ./abo_metadata/ \
    --recursive --no-sign-request
```

**Using ABO models:**

```python
import json
from pathlib import Path

def load_abo_catalog(metadata_dir: str) -> dict:
    """Load ABO metadata for furniture category lookup."""
    catalog = {}
    meta_path = Path(metadata_dir)

    for json_file in meta_path.glob("*.json"):
        with open(json_file) as f:
            for line in f:
                item = json.loads(line)
                item_id = item.get("item_id", "")
                catalog[item_id] = {
                    "category": item.get("product_type", [{}])[0].get("value", "unknown"),
                    "dimensions": item.get("item_dimensions", {}),
                    "model_path": f"abo_models/{item_id}.glb",
                }

    return catalog
```

### Quick lookup — is the catalog already available?

```bash
if [ -n "$FURNITURE_CATALOG_PATH" ] && [ -d "$FURNITURE_CATALOG_PATH" ]; then
    echo "catalog at $FURNITURE_CATALOG_PATH"
    ls "$FURNITURE_CATALOG_PATH/model_info.json" 2>/dev/null
else
    echo "catalog not available; exporters will fall back to AABB cubes"
fi
```

Resolve furniture meshes as `$FURNITURE_CATALOG_PATH/<model_id>/raw_model.obj` when set. The skill's GLTF/XML exporters do this automatically via `FurnitureItem.get_mesh_path()`; fallback boxes are emitted when the path doesn't resolve.

### 3D-FUTURE

**Overview:**
- 16,563 3D furniture models in OBJ format
- Higher geometric quality than ABO (designed for interior design)
- Categories: sofa, chair, table, bed, cabinet, shelf, desk, nightstand, etc.
- Requires manual download (registration at [3D-FUTURE website](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future))

**Download:**
1. Register at the 3D-FUTURE challenge page
2. Request dataset access
3. Download the ZIP archive (~50 GB)
4. Extract to a local directory

**Directory structure:**

```
3D-FUTURE-model/
  <model_id>/
    raw_model.obj       # 3D mesh
    raw_model.mtl       # Material definition
    texture/
      texture_0.jpg     # Diffuse texture
      texture_1.jpg     # (optional) Normal map
  model_info.json       # Category labels, dimensions for all models
```

**Using 3D-FUTURE models:**

```python
import json

def load_3dfuture_catalog(model_info_path: str) -> dict:
    """Load 3D-FUTURE model catalog."""
    with open(model_info_path) as f:
        items = json.load(f)

    catalog = {}
    for item in items:
        model_id = item["model_id"]
        catalog[model_id] = {
            "category": item.get("category", "unknown"),
            "super_category": item.get("super-category", "unknown"),
            "style": item.get("style", "unknown"),
            "dimensions": {
                "length": item.get("length", 0),
                "width": item.get("width", 0),
                "height": item.get("height", 0),
            },
            "model_path": f"3D-FUTURE-model/{model_id}/raw_model.obj",
        }

    return catalog
```

**Test gating pattern (project convention):**

```python
import pytest

def test_furniture_loading():
    model_path = Path("3D-FUTURE-model/some_id/raw_model.obj")
    try:
        mesh = load_obj(model_path)
    except FileNotFoundError:
        pytest.skip("3D-FUTURE dataset not available locally")
```

### Category Normalization

Both catalogs use different naming conventions. Normalize to a canonical set:

```python
CATEGORY_ALIASES = {
    # ABO / 3D-FUTURE name -> canonical name
    "couch": "sofa",
    "loveseat": "sofa",
    "sectional": "sofa",
    "armchair": "chair",
    "office_chair": "chair",
    "dining_chair": "chair",
    "stool": "chair",
    "desk": "table",
    "dining_table": "table",
    "coffee_table": "table",
    "side_table": "table",
    "end_table": "table",
    "nightstand": "table",
    "bedside_table": "table",
    "bookshelf": "shelf",
    "bookcase": "shelf",
    "display_shelf": "shelf",
    "wardrobe": "cabinet",
    "dresser": "cabinet",
    "chest": "cabinet",
    "tv_stand": "cabinet",
    "media_console": "cabinet",
    "single_bed": "bed",
    "double_bed": "bed",
    "bunk_bed": "bed",
    "king_bed": "bed",
    "queen_bed": "bed",
}

def normalize_category(raw_category: str) -> str:
    """Map a raw furniture category to a canonical name."""
    key = raw_category.lower().strip().replace(" ", "_").replace("-", "_")
    return CATEGORY_ALIASES.get(key, key)
```

---

## Data Source Selection Guide

```
What do you need?

Indoor scene geometry?
  -> Have existing floorplan? -> Parse it manually or from image
  -> Want procedurally generated? -> Meta Aria ASE (100K scenes)
  -> Want from real building? -> Measure or use architectural CAD

Outdoor / urban geometry?
  -> Quick flat buildings? -> osmnx programmatic (see also blender-osm-pipeline.md)
  -> Detailed 3D buildings? -> Blender + blender-osm addon
  -> Rural / open area? -> osmnx for roads + simple terrain

Furniture 3D models?
  -> Need permissive license? -> Amazon Berkeley Objects (ABO)
  -> Need highest quality? -> 3D-FUTURE
  -> Need both? -> Use 3D-FUTURE for primary, ABO for gap-filling
```

## Related

- [sionna-scene-editing.md](sionna-scene-editing.md) — importing data into Sionna scenes
- [defaults.md](defaults.md) — default building heights and catalog fallbacks
