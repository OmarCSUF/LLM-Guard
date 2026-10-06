"""
Shared-model check
==================
Loads Layer 2 and Layer 3 together and confirms Layer 3 reuses the Layer-2 Qwen
instance instead of loading a second copy. Prints GPU memory after each load.

Exits 0 when the model is shared, 1 otherwise.

Usage:
    python scripts/check_shared_model.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))  # local checkout without `pip install -e .`

import torch

from llmguard.utils.config_loader import load_config
from llmguard.layer2 import Layer2Judge
from llmguard.layer3 import Layer3Mitigator


def gib(n_bytes: int) -> str:
    return f"{n_bytes / 1024 ** 3:.2f} GiB"


def main() -> int:
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    if not torch.cuda.is_available():
        print("CUDA is not available; this check needs the GPU.")
        return 1

    cfg = load_config()

    judge = Layer2Judge.from_config(cfg)
    after_l2 = torch.cuda.memory_allocated()

    mitigator = Layer3Mitigator.from_config(cfg, load_guard_llm=True, share_from=judge)
    after_l3 = torch.cuda.memory_allocated()

    guard = mitigator.guard_llm
    shared = guard is not None and guard.model is judge.checker.judge._model
    free, total = torch.cuda.mem_get_info()

    print(f"after Layer 2 : {gib(after_l2)}")
    print(f"after Layer 3 : {gib(after_l3)}  (+{gib(after_l3 - after_l2)})")
    print(f"GPU free      : {gib(free)} of {gib(total)}")
    print(f"guard loaded  : {guard is not None}")
    print(f"shared model  : {shared}")
    return 0 if shared else 1


if __name__ == "__main__":
    sys.exit(main())
