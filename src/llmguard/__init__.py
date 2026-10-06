"""LLM-Guard: a calibrated three-layer cascade for prompt-level harm detection.

Sub-packages: layer1 (calibrated detector), layer2 (selective judge), layer3 (mitigation),
evaluation (pipeline evaluator and metrics), redteam (attack suite), utils (config, logging,
provenance). The scripts under scripts/ are the experiment entry points.
"""
__version__ = "1.0.0"
