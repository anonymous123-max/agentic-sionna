"""TX position and target coverage optimization.

Provides gradient-based optimization of transmitter position within bounds
and targeted coverage optimization at specific positions.
"""

from typing import List, Optional, TYPE_CHECKING

try:
    import tensorflow as tf
    import sionna.rt as rt
    from sionna.rt import Receiver, PathSolver

    HAS_SIONNA = True
except ImportError:
    HAS_SIONNA = False
    tf = None
    rt = None
    Receiver = None
    PathSolver = None

if TYPE_CHECKING:
    import tensorflow as tf
    import sionna.rt as rt

from src.wireless.optimization import (
    OptimizationConfig,
    OptimizationResult,
    _check_sionna,
    _create_optimizer,
    optimize_tx_orientation,
)


def optimize_tx_position(
    scene: "rt.Scene",
    bounds: tuple,
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    """Optimize TX position within bounds to maximize coverage.

    Uses gradient descent through differentiable ray tracing to find the
    TX position that maximizes signal strength at configured receivers.

    Args:
        scene: Sionna RT Scene with configured TX and RX
        bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z)) position bounds
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

    # Parse bounds - ensure they are 1D tensors
    min_bounds, max_bounds = bounds
    min_bounds = tf.constant(list(min_bounds), dtype=tf.float32)
    max_bounds = tf.constant(list(max_bounds), dtype=tf.float32)

    # Get the first transmitter
    tx_name = list(scene.transmitters.keys())[0]
    tx = scene.get(tx_name)

    # Get initial position - convert to flat list [x, y, z]
    initial_position = tx.position
    if initial_position is None:
        # Default to center of bounds
        initial_position = ((min_bounds + max_bounds) / 2).numpy().tolist()
    else:
        # Ensure we have a flat list [x, y, z]
        try:
            initial_position = list(tf.reshape(initial_position, [-1]).numpy())
        except Exception:
            initial_position = list(initial_position)

    # Create trainable position variable (1D shape)
    position = tf.Variable(initial_position, dtype=tf.float32)

    # Create optimizer
    optimizer = _create_optimizer(config.optimizer_type, config.learning_rate)

    history = []
    initial_loss = None

    for i in range(config.num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(position)

            # Clip position to bounds
            clipped_position = tf.clip_by_value(position, min_bounds, max_bounds)

            # Update TX position
            tx.position = clipped_position

            # Compute paths using PathSolver (Sionna 1.2.1+ API)
            solver = PathSolver()
            paths = solver(scene, max_depth=config.max_depth)

            # Objective: minimize negative mean power (maximize power)
            path_gains = paths.a
            power = tf.reduce_mean(tf.abs(path_gains) ** 2)
            loss = -power

        if initial_loss is None:
            initial_loss = float(loss)

        history.append(float(loss))

        # Compute and apply gradients
        grads = tape.gradient(loss, [position])

        if grads[0] is not None:
            optimizer.apply_gradients(zip(grads, [position]))

        # Clip position to bounds after gradient update
        position.assign(tf.clip_by_value(position, min_bounds, max_bounds))

        if config.verbose and i % 10 == 0:
            print(f"Iteration {i}: loss = {float(loss):.6f}, pos = {position.numpy()}")

    # Apply final optimized position
    final_position = position.numpy().tolist()
    tx.position = final_position

    final_loss = float(loss)

    return OptimizationResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        iterations=config.num_iterations,
        converged=(final_loss < initial_loss),
        history=history,
    )


def optimize_for_target_coverage(
    scene: "rt.Scene",
    target_positions: List[List[float]],
    config: Optional[OptimizationConfig] = None,
) -> OptimizationResult:
    """Optimize TX to maximize signal strength at specific target positions.

    Temporarily adds receivers at target positions, runs orientation
    optimization, then removes the temporary receivers.

    This is useful for optimizing coverage in specific areas (e.g.,
    desks, seating areas) rather than uniform coverage.

    Args:
        scene: Sionna RT Scene with configured TX
        target_positions: List of [x, y, z] target positions
        config: Optimization configuration (uses defaults if None)

    Returns:
        OptimizationResult with initial/final loss and convergence status

    Raises:
        ImportError: If sionna is not installed
        ValueError: If scene has no transmitters or target_positions is empty
    """
    _check_sionna()

    if not target_positions:
        raise ValueError("target_positions cannot be empty")

    # Validate scene has TX
    if not scene.transmitters:
        raise ValueError("Scene has no transmitters. Use configure_transmitter() first.")

    # Store existing receivers to restore later
    existing_rx_names = list(scene.receivers.keys())

    # Add temporary receivers at target positions
    temp_rx_names = []
    for i, pos in enumerate(target_positions):
        name = f"_target_rx_{i}"
        temp_rx_names.append(name)
        rx = Receiver(name=name, position=pos)
        scene.add(rx)

    # Run orientation optimization
    result = optimize_tx_orientation(scene, config)

    # Remove temporary receivers
    for name in temp_rx_names:
        scene.remove(name)

    return result
