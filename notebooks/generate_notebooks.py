"""Generate Colab notebooks for ResQ-AI experiments.

Creates ready-to-run .ipynb notebooks that clone the repo,
install dependencies, and run the full experiments.
"""

import json
from pathlib import Path


def make_cell(cell_type: str, source: list[str], metadata: dict | None = None) -> dict:
    """Create a notebook cell."""
    cell = {
        "cell_type": cell_type,
        "metadata": metadata or {},
        "source": source,
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def create_colab_smoke_notebook() -> dict:
    """Create smoke test Colab notebook."""
    cells = [
        make_cell("markdown", [
            "# ResQ-AI: Smoke Test Notebook\n",
            "\n",
            "This notebook runs the complete ResQ-AI pipeline in smoke mode.\n",
            "It uses synthetic data and completes in under 5 minutes on CPU.\n",
            "\n",
            "**No GPU required for smoke tests.**"
        ]),
        make_cell("markdown", ["## 1. Setup"]),
        make_cell("code", [
            "# Clone the repository\n",
            "!git clone https://github.com/resq-ai/resq-ai.git 2>/dev/null || true\n",
            "%cd resq-ai"
        ]),
        make_cell("code", [
            "# Install dependencies\n",
            "!pip install -q torch torchvision timm numpy scipy pyyaml tqdm pytest \\\n",
            "    scikit-learn matplotlib seaborn"
        ]),
        make_cell("markdown", ["## 2. Verify Installation"]),
        make_cell("code", [
            "import torch\n",
            "import numpy as np\n",
            "print(f'PyTorch: {torch.__version__}')\n",
            "print(f'NumPy: {np.__version__}')\n",
            "print(f'CUDA available: {torch.cuda.is_available()}')\n",
            "if torch.cuda.is_available():\n",
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')"
        ]),
        make_cell("markdown", ["## 3. Run Smoke Tests"]),
        make_cell("code", [
            "# Run pytest smoke tests\n",
            "!python -m pytest tests/test_smoke.py -v -m smoke --tb=short 2>&1 | tail -30"
        ]),
        make_cell("markdown", ["## 4. Quick Training Demo"]),
        make_cell("code", [
            "import sys\n",
            "sys.path.insert(0, '.')\n",
            "\n",
            "from src.data.synthetic import SyntheticFloodDataset, create_synthetic_dataloaders\n",
            "from src.models.resqnet import ResQNet\n",
            "from src.models.losses import BCEDiceLoss\n",
            "from src.utils.seed import set_seed\n",
            "import torch\n",
            "import torch.nn.functional as F\n",
            "\n",
            "# Setup\n",
            "set_seed(42)\n",
            "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print(f'Using device: {device}')\n",
            "\n",
            "# Config\n",
            "config = {\n",
            "    'project': {'seed': 42},\n",
            "    'data': {\n",
            "        'crop_size': 64,\n",
            "        'batch_size': 4,\n",
            "        'num_train': 32,\n",
            "        'num_val': 8,\n",
            "        'num_test': 8,\n",
            "        'modality_dropout': 0.2,\n",
            "    },\n",
            "    'model': {\n",
            "        'sar_channels': 2,\n",
            "        'optical_channels': 13,\n",
            "        'geo_channels': 6,\n",
            "        'rainfall_seq_len': 30,\n",
            "    },\n",
            "}\n",
            "\n",
            "# Create dataloaders\n",
            "loaders = create_synthetic_dataloaders(config, num_workers=0)\n",
            "print(f'Train batches: {len(loaders[\"train\"])}')\n",
            "print(f'Val batches: {len(loaders[\"valid\"])}')"
        ]),
        make_cell("code", [
            "# Build model\n",
            "model = ResQNet(\n",
            "    sar_channels=2,\n",
            "    optical_channels=13,\n",
            "    geo_channels=6,\n",
            "    rainfall_seq_len=30,\n",
            "    rainfall_d_model=32,\n",
            "    rainfall_nhead=2,\n",
            "    rainfall_layers=1,\n",
            "    encoder_name='resnet34',\n",
            "    pretrained=False,\n",
            "    fusion='film_crossattention',\n",
            "    dropout=0.1,\n",
            ").to(device)\n",
            "\n",
            "total_params = sum(p.numel() for p in model.parameters())\n",
            "print(f'Total parameters: {total_params:,}')\n",
            "\n",
            "# Loss and optimizer\n",
            "criterion = BCEDiceLoss()\n",
            "optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)\n",
            "\n",
            "# Quick training loop\n",
            "model.train()\n",
            "for epoch in range(2):\n",
            "    epoch_loss = 0\n",
            "    for batch in loaders['train']:\n",
            "        sar = batch['sar'].to(device)\n",
            "        optical = batch['optical'].to(device)\n",
            "        geo = batch['geo'].to(device)\n",
            "        rainfall = batch['rainfall'].to(device)\n",
            "        label = batch['label'].to(device)\n",
            "        mask = batch['modality_mask'].to(device)\n",
            "        \n",
            "        optimizer.zero_grad()\n",
            "        output = model(sar=sar, optical=optical, geo=geo, rainfall=rainfall, modality_mask=mask)\n",
            "        logits = output['logits']\n",
            "        \n",
            "        # Resize to match label\n",
            "        if logits.shape[-2:] != label.shape[-2:]:\n",
            "            logits = F.interpolate(logits, size=label.shape[-2:], mode='bilinear', align_corners=False)\n",
            "        \n",
            "        loss = criterion(logits, label.unsqueeze(1))\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        epoch_loss += loss.item()\n",
            "    \n",
            "    print(f'Epoch {epoch+1}/2, Loss: {epoch_loss/len(loaders[\"train\"]):.4f}')\n",
            "\n",
            "print('Training complete!')"
        ]),
        make_cell("markdown", ["## 5. Evaluation Demo"]),
        make_cell("code", [
            "from src.evaluation.metrics import FloodMetrics\n",
            "from src.models.resqnet import mc_dropout_predict\n",
            "\n",
            "# Evaluate on test set\n",
            "model.eval()\n",
            "all_probs = []\n",
            "all_labels = []\n",
            "\n",
            "for batch in loaders['test']:\n",
            "    with torch.no_grad():\n",
            "        output = model(\n",
            "            sar=batch['sar'].to(device),\n",
            "            optical=batch['optical'].to(device),\n",
            "            geo=batch['geo'].to(device),\n",
            "            rainfall=batch['rainfall'].to(device),\n",
            "            modality_mask=batch['modality_mask'].to(device),\n",
            "        )\n",
            "        probs = output['probs']\n",
            "        if probs.shape[-2:] != batch['label'].shape[-2:]:\n",
            "            probs = F.interpolate(probs, size=batch['label'].shape[-2:], mode='bilinear', align_corners=False)\n",
            "        all_probs.append(probs.squeeze(1).cpu())\n",
            "        all_labels.append(batch['label'])\n",
            "\n",
            "probs_cat = torch.cat(all_probs)\n",
            "labels_cat = torch.cat(all_labels)\n",
            "\n",
            "metrics = FloodMetrics.compute(probs_cat, labels_cat)\n",
            "print('Test Metrics:')\n",
            "for k, v in metrics.items():\n",
            "    print(f'  {k}: {v:.4f}')"
        ]),
        make_cell("markdown", ["## 6. MC Dropout Uncertainty"]),
        make_cell("code", [
            "# MC Dropout uncertainty estimation\n",
            "batch = next(iter(loaders['test']))\n",
            "result = mc_dropout_predict(\n",
            "    model, num_samples=5,\n",
            "    sar=batch['sar'].to(device),\n",
            "    optical=batch['optical'].to(device),\n",
            "    geo=batch['geo'].to(device),\n",
            "    rainfall=batch['rainfall'].to(device),\n",
            "    modality_mask=batch['modality_mask'].to(device),\n",
            ")\n",
            "\n",
            "print(f'Mean prob range: [{result[\"probs\"].min():.3f}, {result[\"probs\"].max():.3f}]')\n",
            "print(f'Uncertainty (std) range: [{result[\"std\"].min():.3f}, {result[\"std\"].max():.3f}]')\n",
            "print(f'Epistemic uncertainty range: [{result[\"epistemic_uncertainty\"].min():.3f}, {result[\"epistemic_uncertainty\"].max():.3f}]')"
        ]),
        make_cell("markdown", [
            "## 7. Conformal Prediction"
        ]),
        make_cell("code", [
            "from src.uq.conformal import ConformalCalibrator\n",
            "import numpy as np\n",
            "\n",
            "# Calibrate on validation set\n",
            "cal_probs = probs_cat.numpy().flatten()\n",
            "cal_labels = labels_cat.numpy().flatten()\n",
            "\n",
            "calibrator = ConformalCalibrator()\n",
            "calibrator.calibrate(cal_probs, cal_labels, alpha=0.1)\n",
            "print(f'Conformal threshold: {calibrator.threshold:.4f}')\n",
            "\n",
            "# Evaluate coverage\n",
            "test_probs = probs_cat.numpy().flatten()\n",
            "test_labels = labels_cat.numpy().flatten()\n",
            "sets = calibrator.predict_sets(test_probs)\n",
            "coverage = (sets == (test_labels == 1)).mean()\n",
            "print(f'Empirical coverage: {coverage:.4f}')"
        ]),
        make_cell("markdown", [
            "## Done!\n",
            "\n",
            "This notebook demonstrated the core ResQ-AI pipeline:\n",
            "1. Synthetic data generation\n",
            "2. ResQNet model training\n",
            "3. Flood prediction evaluation\n",
            "4. MC Dropout uncertainty estimation\n",
            "5. Conformal prediction calibration\n",
            "\n",
            "For full experiments with real data, see the `experiments/run_experiments.py` script."
        ]),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "name": "ResQ-AI Smoke Test"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "cells": cells,
    }
    return notebook


def create_colab_full_notebook() -> dict:
    """Create full experiment Colab notebook (requires GPU)."""
    cells = [
        make_cell("markdown", [
            "# ResQ-AI: Full Experiments Notebook\n",
            "\n",
            "This notebook runs the complete ResQ-AI experiments on Google Colab.\n",
            "\n",
            "**Requirements:**\n",
            "- GPU runtime (T4 recommended)\n",
            "- ~2-4 hours for full experiments\n",
            "\n",
            "**Change Runtime:**\n",
            "Runtime → Change runtime type → GPU (T4)"
        ]),
        make_cell("code", [
            "# Check GPU\n",
            "!nvidia-smi"
        ]),
        make_cell("code", [
            "# Clone repo and install\n",
            "!git clone https://github.com/resq-ai/resq-ai.git 2>/dev/null || true\n",
            "%cd resq-ai\n",
            "!pip install -q -r requirements.txt"
        ]),
        make_cell("code", [
            "# Run smoke test first to verify setup\n",
            "!python -m pytest tests/test_smoke.py -v -m smoke --tb=short"
        ]),
        make_cell("markdown", ["## Run Experiments"]),
        make_cell("code", [
            "# E1: Main model comparison\n",
            "!python experiments/run_experiments.py --experiment e1 --config configs/default.yaml"
        ]),
        make_cell("code", [
            "# E2: Ablation studies\n",
            "!python experiments/run_experiments.py --experiment e2 --config configs/default.yaml"
        ]),
        make_cell("code", [
            "# E3: Calibration analysis\n",
            "!python experiments/run_experiments.py --experiment e3 --config configs/default.yaml"
        ]),
        make_cell("code", [
            "# E4: Conformal prediction\n",
            "!python experiments/run_experiments.py --experiment e4 --config configs/default.yaml"
        ]),
        make_cell("code", [
            "# E5: Robustness tests\n",
            "!python experiments/run_experiments.py --experiment e5 --config configs/default.yaml"
        ]),
        make_cell("code", [
            "# E6: Decision-level allocation evaluation\n",
            "!python experiments/run_experiments.py --experiment e6 --config configs/default.yaml"
        ]),
        make_cell("code", [
            "# E7: Efficiency benchmarks\n",
            "!python experiments/run_experiments.py --experiment e7 --config configs/default.yaml"
        ]),
        make_cell("markdown", ["## Generate Figures and Tables"]),
        make_cell("code", [
            "!python experiments/generate_figures.py\n",
            "!python experiments/generate_tables.py"
        ]),
        make_cell("code", [
            "# Display key figures\n",
            "from IPython.display import Image, display\n",
            "import glob\n",
            "\n",
            "for fig in sorted(glob.glob('figures/*.png')):\n",
            "    print(f'\\n--- {fig} ---')\n",
            "    display(Image(filename=fig, width=600))"
        ]),
        make_cell("markdown", ["## Download Results"]),
        make_cell("code", [
            "# Zip results for download\n",
            "!zip -r resq_ai_results.zip results/ figures/ paper/tables/\n",
            "\n",
            "from google.colab import files\n",
            "files.download('resq_ai_results.zip')"
        ]),
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "name": "ResQ-AI Full Experiments",
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": cells,
    }
    return notebook


if __name__ == "__main__":
    notebooks_dir = Path(__file__).parent.parent / "notebooks"
    notebooks_dir.mkdir(exist_ok=True)

    # Smoke test notebook
    smoke_nb = create_colab_smoke_notebook()
    smoke_path = notebooks_dir / "colab_smoke_test.ipynb"
    with open(smoke_path, "w") as f:
        json.dump(smoke_nb, f, indent=2)
    print(f"Created: {smoke_path}")

    # Full experiments notebook
    full_nb = create_colab_full_notebook()
    full_path = notebooks_dir / "colab_full_experiments.ipynb"
    with open(full_path, "w") as f:
        json.dump(full_nb, f, indent=2)
    print(f"Created: {full_path}")
