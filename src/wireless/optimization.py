"""Differentiable ray tracing optimization for TX position and orientation.

Enables gradient-based optimization of transmitter placement and orientation
to maximize coverage or signal strength at target locations. This is a key
capability for automated network planning.

Uses TensorFlow's GradientTape with Sionna's differentiable ray tracing to
compute gradients through the path computation, allowing optimization of:
- TX orientation (yaw, pitch, roll) for directional antennas
- TX position within bounds for optimal placement
- Combined orientation and position for full optimization

The optimization minimizes negative received power (maximizes power) at
configured receivers or target positions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

try:
    import tensorflow as tf
    import sionna.rt as rt
    from sionna.rt import Transmitter, Receiver, PathSolver

    HAS_SIONNA = True
except ImportError:
    HAS_SIONNA = False
    tf = None
    rt = None
    Transmitter = None
    Receiver = None
    PathSolver = None

if TYPE_CHECKING:
    import tensorflow as tf
    import sionna.rt as rt


def _check_sionna() -> None:
    """Raise ImportError if sionna is not installed."""
    if not HAS_SIONNA:
        raise ImportError(
            "sionna is required for wireless simulation. "
            "Install with: pip install sionna>=1.2.1"
        )


@dataclass(frozen=True)
class OptimizationConfig:
    """Configuration for TX optimization.

    Attributes:
        num_iterations: Number of optimization iterations (default 100)
        learning_rate: Learning rate for optimizer (default 0.01)
        optimizer_type: Optimizer type - "adam" or "sgd" (default "adam")
        max_depth: Ray tracing depth for optimization (default 3 for speed)
        verbose: Print progress during optimization (default False)
    """

    num_iterations: int = 100
    learning_rate: float = 0.01
    optimizer_type: str = "adam"
    max_depth: int = 3
    verbose: bool = False


@dataclass
class OptimizationResult:
    """Result of TX optimization.

    Attributes:
        initial_loss: Loss value before optimization
        final_loss: Loss value after optimization
        iterations: Number of iterations completed
        converged: True if final_loss < initial_loss (improvement)
        history: Loss value at each iteration
    """

    initial_loss: float
    final_loss: float
    iterations: int
    converged: bool
    history: List[float] = field(default_factory=list)


def _create_optimizer(
    optimizer_type: str, learning_rate: float
) -> "tf.optimizers.Optimizer":
    """Create TensorFlow optimizer based on config.

    Args:
        optimizer_type: "adam" or "sgd"
        learning_rate: Learning rate for optimizer

    Returns:
        Configured TensorFlow optimizer
    """
    if optimizer_type.lower() == "adam":
        return tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type.lower() == "sgd":
        return tf.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}. Use 'adam' or 'sgd'.")


def optimize_tx_orientation(
    scene: "rt.Scene",
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    """Optimize TX orientation to maximize received power at all RX.

    Uses gradient descent through differentiable ray tracing to find the
    TX orientation that maximizes signal strength at configured receivers.

    The orientation is represented as [yaw, pitch, roll] in radians.
    Yaw rotates around Z axis, pitch around Y, roll around X.

    Args:
        scene: Sionna RT Scene with configured TX and RX
        config: Optimization configuration (uses defaults if None)

    Returns:
        OptimizationResult with initial/final loss and convergence status

    Raises:
        ImportError: If sionna is not installed
        ValueError: If scene has no transmitters or receivers
    """
    _check_sionna()

    if config is None:
        config = OptimizationConfig()

    # Validate scene
    if not scene.transmitters:
        raise ValueError("Scene has no transmitters. Use configure_transmitter() first.")
    if not scene.receivers:
        raise ValueError("Scene has no receivers. Use configure_receivers() first.")

    # Get the first transmitter (typical single-TX scenario)
    tx_name = list(scene.transmitters.keys())[0]
    tx = scene.get(tx_name)

    # Get initial orientation
    initial_orientation = tx.orientation
    if initial_orientation is None:
        initial_orientation = [0.0, 0.0, 0.0]

    # Create trainable orientation variable
    orientation = tf.Variable(initial_orientation, dtype=tf.float32)

    # Create optimizer
    optimizer = _create_optimizer(config.optimizer_type, config.learning_rate)

    history = []
    initial_loss = None

    for i in range(config.num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(orientation)

            # Update TX orientation
            tx.orientation = orientation

            # Compute paths using PathSolver (Sionna 1.2.1+ API)
            solver = PathSolver()
            paths = solver(scene, max_depth=config.max_depth)

            # Objective: minimize negative mean power (maximize power)
            # paths.a contains complex path gains
            # Sum over all paths, take magnitude squared for power
            path_gains = paths.a
            power = tf.reduce_mean(tf.abs(path_gains) ** 2)

            # Negative because we minimize (want to maximize power)
            loss = -power

        if initial_loss is None:
            initial_loss = float(loss)

        history.append(float(loss))

        # Compute and apply gradients
        grads = tape.gradient(loss, [orientation])

        # Only apply if gradients exist (may be None if no paths)
        if grads[0] is not None:
            optimizer.apply_gradients(zip(grads, [orientation]))

        if config.verbose and i % 10 == 0:
            print(f"Iteration {i}: loss = {float(loss):.6f}")

    # Apply final optimized orientation
    final_orientation = orientation.numpy().tolist()
    tx.orientation = final_orientation

    final_loss = float(loss)

    return OptimizationResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        iterations=config.num_iterations,
        converged=(final_loss < initial_loss),
        history=history,
    )


# Import from extracted module for backward compatibility
from src.wireless.tx_placement import (  # noqa: F401
    optimize_tx_position,
    optimize_for_target_coverage,
)
