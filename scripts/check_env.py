#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys

import torch


def main() -> None:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"PyTorch: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        return

    n = torch.cuda.device_count()
    print(f"CUDA device count: {n}")
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        major, minor = torch.cuda.get_device_capability(i)
        print(f"- GPU{i}: {name} (cc={major}.{minor})")

    bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    print(f"BF16 supported: {bf16_ok}")

    # Simple sanity check: allocate and matmul.
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if bf16_ok else torch.float16
    a = torch.randn((1024, 1024), device=device, dtype=dtype)
    b = torch.randn((1024, 1024), device=device, dtype=dtype)
    c = a @ b
    torch.cuda.synchronize()
    print(f"Matmul OK: {c.shape} dtype={c.dtype}")


if __name__ == "__main__":
    main()

