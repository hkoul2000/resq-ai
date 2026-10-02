# ResQ-AI: Uncertainty-Aware Multimodal AI Framework for Urban Disaster Prediction

An advanced multimodal machine learning framework designed for predicting and optimizing responses to urban disasters.

## Project Structure

- `src/`: Source code
  - `data/`: Data loading and processing
  - `models/`: ML models
  - `uq/`: Uncertainty quantification
  - `optimization/`: Allocation optimization
  - `evaluation/`: Metrics and evaluation
  - `utils/`: Utilities
  - `impact/`: Impact analysis
- `configs/`: Configuration files
- `tests/`: Unit and integration tests

## Architecture

```
[Data Layer (Multimodal)] -> [Fusion & Modeling] -> [Uncertainty Quantification] -> [Optimization]
```

## Setup

1. Install dependencies:
   ```bash
   make setup
   ```

2. Run smoke tests:
   ```bash
   make smoke
   ```

## Development

Use CPU with `--smoke` flag for rapid testing.
