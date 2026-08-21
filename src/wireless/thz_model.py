"""THz (>100 GHz) theoretical coverage model.

Provides physics-based THz coverage computation with line-of-sight
ray-obstacle intersection for indoor and outdoor scenes. Uses theoretical
path loss calculations since Sionna ITU materials don't support THz frequencies.
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class THzConfig:
    """Configuration for THz (>100 GHz) theoretical coverage model.

    Note: Sionna ITU materials don't support THz frequencies. This model uses
    theoretical path loss calculations with furniture blockage effects.

    Attributes:
        frequency: Carrier frequency in Hz (default 300 GHz)
        tx_power_dbm: Transmitter power in dBm (default 30 dBm = 1W)
        tx_position: [x, y, z] position of transmitter in meters
        rx_height: Height of receiver plane in meters (default 1.0)
        array_elements: Number of antenna elements (default 256 for 16x16)
        resolution: Grid resolution in meters (default 0.1)
        atmospheric_absorption: dB per meter (default 0.15 for 300 GHz)
        furniture_attenuation_per_meter: dB per meter of furniture (default 30)
    """

    frequency: float = 300e9  # 300 GHz THz
    tx_power_dbm: float = 30.0
    tx_position: Tuple[float, float, float] = (6.0, 5.0, 2.5)
    rx_height: float = 1.0
    array_elements: int = 256  # 16x16 massive MIMO
    resolution: float = 0.1
    atmospheric_absorption: float = 0.15  # dB/m at 300 GHz
    furniture_attenuation_per_meter: float = 30.0  # dB/m through wood


def _ray_intersects_aabb_2d(
    ox: float, oy: float,
    dx: float, dy: float,
    xmin: float, ymin: float,
    xmax: float, ymax: float,
    max_t: float,
) -> bool:
    """Fast 2D ray-AABB intersection test (slab method).

    Returns True if the ray from (ox,oy) in direction (dx,dy)
    hits the box [xmin,ymin]-[xmax,ymax] at t < max_t.
    """
    tmin = 0.0
    tmax = max_t

    for o, d, bmin, bmax in [(ox, dx, xmin, xmax), (oy, dy, ymin, ymax)]:
        if abs(d) < 1e-12:
            if o < bmin or o > bmax:
                return False
            continue
        t1 = (bmin - o) / d
        t2 = (bmax - o) / d
        if t1 > t2:
            t1, t2 = t2, t1
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return True


def compute_thz_coverage(
    room_width: float,
    room_length: float,
    furniture_items: list,
    config: Optional[THzConfig] = None,
    obstacles: list = None,
) -> "np.ndarray":
    """Compute coverage map with LOS ray-obstacle intersection.

    Uses physics-based model with actual line-of-sight checks against
    obstacle AABBs (buildings and furniture). For each grid cell, a ray
    is cast from the TX; if it intersects an obstacle, penetration loss
    is applied per obstacle crossed.

    Model:
    - Free space path loss (FSPL)
    - Massive MIMO array gain
    - Atmospheric absorption
    - LOS obstruction: ray-AABB intersection against all obstacles
    - Each blocked obstacle adds material-dependent penetration loss

    Args:
        room_width: Scene width in meters
        room_length: Scene length in meters
        furniture_items: List of FurnitureItem objects (indoor furniture)
        config: THz configuration (uses defaults if None)
        obstacles: List of dicts with keys xmin,xmax,ymin,ymax,zmin,zmax,
                   loss_db (optional, default 20 dB per obstacle).
                   Use this for buildings in outdoor scenes.

    Returns:
        2D numpy array of signal strength in dBm
    """
    import numpy as np

    if config is None:
        config = THzConfig()
    if obstacles is None:
        obstacles = []

    # Build unified AABB obstacle list: (xmin, ymin, xmax, ymax, height, loss_db)
    aabbs = []

    # Buildings / explicit obstacles
    for ob in obstacles:
        aabbs.append((
            float(ob["xmin"]), float(ob["ymin"]),
            float(ob["xmax"]), float(ob["ymax"]),
            float(ob.get("zmax", 30)),
            float(ob.get("loss_db", 25)),  # concrete ~25 dB penetration
        ))

    # Furniture items
    for item in furniture_items:
        ix, iy = item.position.x, item.position.y
        w = item.dimensions.width
        dep = item.dimensions.depth
        h = item.dimensions.height
        if h > config.rx_height * 0.5:  # only if tall enough to matter
            aabbs.append((
                ix - w / 2, iy - dep / 2,
                ix + w / 2, iy + dep / 2,
                h,
                min(config.furniture_attenuation_per_meter * (w + dep) / 2, 30),
            ))

    # Create grid
    nx = max(2, int(room_width / config.resolution))
    ny = max(2, int(room_length / config.resolution))
    x = np.linspace(0, room_width, nx)
    y = np.linspace(0, room_length, ny)
    X, Y = np.meshgrid(x, y)

    tx_x, tx_y, tx_z = config.tx_position

    # Distance from TX to each grid point
    d = np.sqrt((X - tx_x)**2 + (Y - tx_y)**2 + (config.rx_height - tx_z)**2)
    d = np.maximum(d, 0.1)

    # Free space path loss
    fspl_db = 20 * np.log10(d) + 20 * np.log10(config.frequency) - 147.55

    # Array gain from massive MIMO
    array_gain_db = 10 * np.log10(config.array_elements)

    # Atmospheric absorption
    atm_loss = config.atmospheric_absorption * d

    # Base coverage
    coverage = config.tx_power_dbm + array_gain_db - fspl_db - atm_loss

    # LOS obstruction via ray-AABB intersection
    if aabbs:
        obstruction = np.zeros_like(coverage)
        for j in range(ny):
            for i in range(nx):
                rx_x, rx_y = x[i], y[j]
                ray_dx = rx_x - tx_x
                ray_dy = rx_y - tx_y
                ray_len = np.sqrt(ray_dx**2 + ray_dy**2)
                if ray_len < 0.01:
                    continue
                # Normalize direction
                rdx = ray_dx / ray_len
                rdy = ray_dy / ray_len

                for xmin, ymin, xmax, ymax, ob_h, loss in aabbs:
                    # Height check: obstacle must be tall enough to block at rx_height
                    if ob_h < config.rx_height * 0.7:
                        continue
                    if _ray_intersects_aabb_2d(
                        tx_x, tx_y, rdx, rdy,
                        xmin, ymin, xmax, ymax, ray_len
                    ):
                        obstruction[j, i] += loss

        # Cap total obstruction (multiple walls = diminishing returns)
        obstruction = np.minimum(obstruction, 60)
        coverage -= obstruction

    return coverage
