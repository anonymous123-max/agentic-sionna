"""Generate mock coverage maps with deliberate problems for diagnose tasks."""
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent


def kitchen_dead_spot():
    """10x8 m room (2D), 0.5m grid, AP at (5,4). RSS in dBm.
    Deliberate dead spot at (8,7) corner: -90 dBm vs expected ~-65."""
    grid = np.full((20, 16), -65.0)  # baseline good coverage
    # Add a rectangular dead zone in the kitchen corner (bottom-right)
    grid[16:20, 14:16] = -90.0  # wall-occlusion / material-penetration dead zone
    return grid


def hallway_propagation_loss():
    """Long corridor: linear path-loss but 20 dB worse than FSPL prediction."""
    distances = np.arange(1, 31).astype(float)  # 1m to 30m
    fspl = 20 * np.log10(distances) + 32.4 + 20 * np.log10(3.5)  # 3.5 GHz
    measured = fspl + 20.0  # +20 dB extra loss (material penetration / wall bounces)
    return np.column_stack([distances, fspl, measured])


def channel_estimation_diverges():
    """200 SGD steps where MSE doesn't converge (stuck around 1.0)."""
    losses = 1.0 + 0.1 * np.random.RandomState(42).randn(200)
    return np.maximum(losses, 0.01)


if __name__ == "__main__":
    np.save(OUT / "kitchen_dead_spot.npy", kitchen_dead_spot())
    np.save(OUT / "hallway_propagation_loss.npy", hallway_propagation_loss())
    np.save(OUT / "channel_estimation_diverges.npy", channel_estimation_diverges())
    print("Generated 3 diagnose fixtures")
    for f in sorted(OUT.glob("*.npy")):
        arr = np.load(f)
        print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}, size={f.stat().st_size}B")
