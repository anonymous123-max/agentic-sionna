"""Make scripts/ and the repo root importable for tests in this directory."""
import sys
from pathlib import Path
# scripts/ — for verify_output and other skill scripts
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# repo root — so `benchmark._verifier_core` is importable after D.5 move
sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
