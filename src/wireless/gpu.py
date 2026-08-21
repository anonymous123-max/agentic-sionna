"""GPU compatibility utilities for RTX 5090 (Blackwell sm_120).

RTX 5090 GPUs require CUDA 12.8+ and TensorFlow nightly builds with
sm_120 support. This module provides detection and setup utilities
to ensure compatibility before running wireless simulations.

Usage:
    # Call BEFORE importing TensorFlow
    from src.wireless.gpu import setup_rtx5090_environment
    setup_rtx5090_environment()

    # Then import TensorFlow
    import tensorflow as tf

    # Verify GPU is working
    from src.wireless.gpu import verify_gpu_capability
    success, message = verify_gpu_capability()
    if not success:
        print(f"GPU issue: {message}")
"""

import os
import re
import subprocess
import sys
from typing import Tuple, Dict, Any


def setup_rtx5090_environment() -> None:
    """Configure environment variables for RTX 5090 compatibility.

    Sets:
    - TORCH_CUDA_ARCH_LIST=sm_120 for PyTorch Blackwell support
    - TF_FORCE_GPU_ALLOW_GROWTH=true to avoid OOM on large VRAM GPUs

    IMPORTANT: Must be called BEFORE importing TensorFlow or PyTorch.
    """
    # Check if TensorFlow already imported
    if "tensorflow" in sys.modules:
        print(
            "Warning: TensorFlow already imported. Environment variables "
            "may not take effect. Call setup_rtx5090_environment() before "
            "importing tensorflow.",
            file=sys.stderr,
        )

    # Check if PyTorch already imported
    if "torch" in sys.modules:
        print(
            "Warning: PyTorch already imported. TORCH_CUDA_ARCH_LIST "
            "may not take effect. Call setup_rtx5090_environment() before "
            "importing torch.",
            file=sys.stderr,
        )

    # Set environment variables for Blackwell architecture
    os.environ["TORCH_CUDA_ARCH_LIST"] = "sm_120"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def check_cuda_version() -> Tuple[bool, str]:
    """Check if CUDA toolkit version is 12.8+ (required for sm_120).

    Returns:
        Tuple of (is_compatible, message):
        - (True, "12.8.1") if CUDA 12.8+ found
        - (False, "CUDA 12.3 found, requires 12.8+") if version too old
        - (False, "nvcc not found...") if CUDA toolkit not installed
    """
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = result.stdout + result.stderr

        # Parse version from output like "release 12.8, V12.8.93"
        version_match = re.search(r"release (\d+)\.(\d+)", output)
        if not version_match:
            return (False, f"Could not parse CUDA version from: {output[:100]}")

        major = int(version_match.group(1))
        minor = int(version_match.group(2))
        version_str = f"{major}.{minor}"

        # RTX 5090 (sm_120) requires CUDA 12.8+
        if major > 12 or (major == 12 and minor >= 8):
            return (True, version_str)
        else:
            return (
                False,
                f"CUDA {version_str} found, requires 12.8+ for RTX 5090 (sm_120)",
            )

    except FileNotFoundError:
        return (
            False,
            "nvcc not found. Install CUDA Toolkit 12.8+ from "
            "https://developer.nvidia.com/cuda-downloads",
        )
    except subprocess.TimeoutExpired:
        return (False, "nvcc command timed out")
    except Exception as e:
        return (False, f"Error checking CUDA version: {e}")


def verify_gpu_capability() -> Tuple[bool, str]:
    """Verify TensorFlow can use GPU for computation.

    Attempts to:
    1. List available GPU devices
    2. Run a simple matmul operation on GPU

    Returns:
        Tuple of (success, message):
        - (True, "NVIDIA GeForce RTX 5090") if GPU works
        - (False, "No GPU detected") if no GPU available
        - (False, "sm_120 is not compatible...") if kernel compatibility issue
    """
    try:
        import tensorflow as tf
    except ImportError:
        return (
            False,
            "TensorFlow not installed. Install with: pip install --pre tensorflow",
        )

    try:
        # Check for GPU devices
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            return (False, "No GPU detected by TensorFlow")

        # Get GPU name from first device
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        gpu_name = gpu_details.get("device_name", "Unknown GPU")

        # Attempt simple GPU operation to verify kernels work
        with tf.device("/GPU:0"):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            _ = tf.matmul(a, b)

        return (True, gpu_name)

    except RuntimeError as e:
        error_msg = str(e)
        # Check for common sm_120 compatibility errors
        if "sm_120" in error_msg or "INVALID_PTX" in error_msg.upper():
            return (
                False,
                f"sm_120 kernel not available. Install TensorFlow nightly: "
                f"pip install --pre tensorflow. Error: {error_msg[:200]}",
            )
        return (False, f"GPU runtime error: {error_msg[:200]}")
    except Exception as e:
        return (False, f"GPU verification failed: {e}")


def get_gpu_info() -> Dict[str, Any]:
    """Get comprehensive GPU and environment information.

    Returns:
        Dict with keys:
        - gpu_name: str or None
        - gpu_available: bool
        - cuda_version: str or None
        - cuda_compatible: bool
        - tensorflow_version: str or None
        - tensorflow_available: bool
        - is_rtx5090_compatible: bool (all requirements met)
        - issues: list of str (problems found)
    """
    info: Dict[str, Any] = {
        "gpu_name": None,
        "gpu_available": False,
        "cuda_version": None,
        "cuda_compatible": False,
        "tensorflow_version": None,
        "tensorflow_available": False,
        "is_rtx5090_compatible": False,
        "issues": [],
    }

    # Check CUDA version
    cuda_ok, cuda_msg = check_cuda_version()
    info["cuda_compatible"] = cuda_ok
    if cuda_ok:
        info["cuda_version"] = cuda_msg
    else:
        info["issues"].append(cuda_msg)

    # Check TensorFlow
    try:
        import tensorflow as tf

        info["tensorflow_available"] = True
        info["tensorflow_version"] = tf.__version__
    except ImportError:
        info["issues"].append(
            "TensorFlow not installed. Install with: pip install --pre tensorflow"
        )

    # Check GPU capability (only if TensorFlow available)
    if info["tensorflow_available"]:
        gpu_ok, gpu_msg = verify_gpu_capability()
        info["gpu_available"] = gpu_ok
        if gpu_ok:
            info["gpu_name"] = gpu_msg
        else:
            info["issues"].append(gpu_msg)

    # Overall RTX 5090 compatibility
    info["is_rtx5090_compatible"] = (
        info["cuda_compatible"]
        and info["tensorflow_available"]
        and info["gpu_available"]
    )

    return info
