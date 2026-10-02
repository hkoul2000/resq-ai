# ResQ-AI: Design Decisions Log

## D001: Python Version
**Decision**: Use Python 3.13 (available on dev machine) with compatibility target 3.11+
**Reasoning**: Spec requires 3.11, but 3.13 is available locally. Docker uses 3.11. All code written to be compatible with 3.11+.
**Date**: 2026-10-02

## D002: No Local GPU
**Decision**: All development done CPU-only with --smoke mode. Full experiments via Colab notebooks.
**Reasoning**: Dev machine has no GPU. Code designed with T4 16GB target. Smoke tests validate logic on CPU.
**Date**: 2026-10-02
