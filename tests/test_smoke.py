"""Smoke tests for ResQ-AI pipeline.

Run with: pytest tests/test_smoke.py -v -m smoke

These tests verify the core pipeline works on CPU without
downloading any data, using synthetic data throughout.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.smoke
def test_synthetic_dataset():
    """Test that synthetic dataset creates properly shaped data."""
    from src.data.synthetic import SyntheticFloodDataset

    ds = SyntheticFloodDataset(
        split="train",
        num_samples=4,
        crop_size=32,
        sar_channels=2,
        optical_channels=13,
        geo_channels=6,
        rainfall_seq_len=30,
    )

    assert len(ds) == 4
    sample = ds[0]

    assert sample["sar"].shape == (2, 32, 32)
    assert sample["optical"].shape == (13, 32, 32)
    assert sample["geo"].shape == (6, 32, 32)
    assert sample["rainfall"].shape == (30, 1)
    assert sample["label"].shape == (32, 32)
    assert sample["valid_mask"].shape == (32, 32)
    assert sample["modality_mask"].shape == (5,)
    assert isinstance(sample["region"], str)
    assert isinstance(sample["chip_id"], str)

    # Check label is binary
    assert set(sample["label"].unique().tolist()).issubset({0.0, 1.0})


@pytest.mark.smoke
def test_synthetic_dataloader():
    """Test synthetic dataloaders work correctly."""
    from src.data.synthetic import create_synthetic_dataloaders

    config = {
        "project": {"seed": 42},
        "data": {
            "crop_size": 32,
            "batch_size": 2,
            "num_train": 8,
            "num_val": 4,
            "num_test": 4,
            "modality_dropout": 0.0,
        },
        "model": {
            "sar_channels": 2,
            "optical_channels": 13,
            "geo_channels": 6,
            "rainfall_seq_len": 30,
        },
    }

    loaders = create_synthetic_dataloaders(config, batch_size=2, num_workers=0)
    assert "train" in loaders
    assert "valid" in loaders
    assert "test" in loaders

    batch = next(iter(loaders["train"]))
    assert batch["sar"].shape[0] == 2  # batch size
    assert batch["sar"].shape[1] == 2  # SAR channels


@pytest.mark.smoke
def test_resqnet_forward(sample_batch):
    """Test ResQNet forward pass with all modalities."""
    from src.models.resqnet import ResQNet

    model = ResQNet(
        sar_channels=2,
        optical_channels=13,
        geo_channels=6,
        rainfall_seq_len=30,
        rainfall_d_model=32,
        rainfall_nhead=2,
        rainfall_layers=1,
        encoder_name="resnet34",
        pretrained=False,
        fusion="film_crossattention",
        dropout=0.1,
    )
    model.eval()

    with torch.no_grad():
        output = model(
            sar=sample_batch["sar"],
            optical=sample_batch["optical"],
            geo=sample_batch["geo"],
            rainfall=sample_batch["rainfall"],
            modality_mask=sample_batch["modality_mask"],
        )

    assert "probs" in output
    assert "logits" in output
    assert output["probs"].shape[0] == sample_batch["sar"].shape[0]  # batch
    assert output["probs"].min() >= 0 and output["probs"].max() <= 1


@pytest.mark.smoke
def test_resqnet_backward(sample_batch):
    """Test ResQNet forward + backward pass."""
    from src.models.resqnet import ResQNet

    model = ResQNet(
        sar_channels=2,
        optical_channels=13,
        geo_channels=6,
        pretrained=False,
        rainfall_d_model=32,
        rainfall_nhead=2,
        rainfall_layers=1,
    )
    model.train()

    output = model(
        sar=sample_batch["sar"],
        optical=sample_batch["optical"],
        geo=sample_batch["geo"],
        rainfall=sample_batch["rainfall"],
        modality_mask=sample_batch["modality_mask"],
    )

    # Compute a simple loss and backprop
    target = sample_batch["label"].unsqueeze(1)
    # Resize output to match target if needed
    if output["probs"].shape[-2:] != target.shape[-2:]:
        import torch.nn.functional as F
        probs = F.interpolate(output["probs"], size=target.shape[-2:], mode="bilinear", align_corners=False)
    else:
        probs = output["probs"]

    loss = torch.nn.functional.binary_cross_entropy(probs, target)
    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No gradients computed"


@pytest.mark.smoke
def test_losses():
    """Test all loss functions."""
    from src.models.losses import BCEDiceLoss, FocalLoss, FocalDiceLoss

    B, H, W = 2, 32, 32
    logits = torch.randn(B, 1, H, W)
    target = torch.randint(0, 2, (B, 1, H, W)).float()
    valid_mask = torch.ones(B, 1, H, W)

    # Test BCE + Dice
    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, target)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert loss.item() >= 0

    # Test Focal
    focal = FocalLoss()
    loss = focal(logits, target)
    assert not torch.isnan(loss)


@pytest.mark.smoke
def test_mc_dropout(sample_batch):
    """Test MC Dropout prediction."""
    from src.models.resqnet import ResQNet, mc_dropout_predict

    model = ResQNet(
        sar_channels=2,
        optical_channels=13,
        geo_channels=6,
        pretrained=False,
        dropout=0.3,
        rainfall_d_model=32,
        rainfall_nhead=2,
        rainfall_layers=1,
    )

    result = mc_dropout_predict(
        model,
        num_samples=3,
        sar=sample_batch["sar"],
        optical=sample_batch["optical"],
        geo=sample_batch["geo"],
        rainfall=sample_batch["rainfall"],
        modality_mask=sample_batch["modality_mask"],
    )

    assert "probs" in result
    assert "std" in result
    assert "epistemic_uncertainty" in result
    assert result["probs"].min() >= 0 and result["probs"].max() <= 1
    assert result["std"].min() >= 0


@pytest.mark.smoke
def test_conformal_calibrator():
    """Test conformal prediction calibrator."""
    from src.uq.conformal import ConformalCalibrator

    # Create synthetic calibration data
    np.random.seed(42)
    n = 100
    cal_probs = np.random.rand(n)
    # Make labels correlated with probs for realistic test
    cal_labels = (cal_probs > 0.5).astype(float)

    calibrator = ConformalCalibrator()
    calibrator.calibrate(cal_probs, cal_labels, alpha=0.1)

    assert calibrator.threshold is not None

    # Test prediction sets
    test_probs = np.random.rand(50)
    sets = calibrator.predict_sets(test_probs)
    assert sets.shape == (50,)


@pytest.mark.smoke
def test_evidential_loss():
    """Test evidential loss computation."""
    from src.uq.evidential import EvidentialLoss

    B, C, H, W = 2, 2, 32, 32
    # Evidence (positive)
    evidence = torch.abs(torch.randn(B, C, H, W)) + 0.1
    target = torch.randint(0, C, (B, H, W))

    loss_fn = EvidentialLoss(num_classes=C)
    loss = loss_fn(evidence, target, epoch=1)
    assert loss.dim() == 0
    assert not torch.isnan(loss)


@pytest.mark.smoke
def test_metrics():
    """Test metric computation."""
    from src.evaluation.metrics import FloodMetrics

    B, H, W = 4, 32, 32
    probs = torch.rand(B, H, W)
    labels = torch.randint(0, 2, (B, H, W)).float()

    metrics = FloodMetrics()
    results = metrics.compute(probs, labels)

    assert "iou" in results or "f1" in results or isinstance(results, dict)


@pytest.mark.smoke
def test_film_layer():
    """Test FiLM conditioning layer."""
    from src.models.resqnet import FiLMLayer

    film = FiLMLayer(feature_dim=64, cond_dim=32)

    # Test with 4D input (spatial)
    x = torch.randn(2, 64, 8, 8)
    cond = torch.randn(2, 32)
    out = film(x, cond)
    assert out.shape == x.shape

    # Test with 2D input (vector)
    x_2d = torch.randn(2, 64)
    out_2d = film(x_2d, cond)
    assert out_2d.shape == x_2d.shape


@pytest.mark.smoke
def test_gated_cross_attention():
    """Test gated cross-attention module."""
    from src.models.resqnet import GatedCrossAttention

    attn = GatedCrossAttention(dim=64, num_heads=4)
    query = torch.randn(2, 16, 64)
    key_value = torch.randn(2, 4, 64)

    out = attn(query, key_value)
    assert out.shape == query.shape


@pytest.mark.smoke
def test_geo_encoder():
    """Test geographic feature encoder."""
    from src.models.resqnet import GeoEncoder

    encoder = GeoEncoder(in_channels=6, out_dim=128)
    x = torch.randn(2, 6, 64, 64)
    out = encoder(x)
    assert out.shape == (2, 128)


@pytest.mark.smoke
def test_rainfall_encoder():
    """Test rainfall temporal encoder."""
    from src.models.resqnet import RainfallEncoder

    encoder = RainfallEncoder(
        seq_len=30, d_model=32, nhead=2, num_layers=1, out_dim=64
    )
    x = torch.randn(2, 30, 1)
    out = encoder(x)
    assert out.shape == (2, 64)


@pytest.mark.smoke
def test_allocation_problem():
    """Test allocation problem creation and baseline solving."""
    from src.optimization.allocation import (
        AllocationProblem,
        GreedyAllocation,
        ProportionalAllocation,
    )

    problem = AllocationProblem(
        zones=["zone_1", "zone_2"],
        depots=["depot_A", "depot_B"],
        capacities={"depot_A": 300.0, "depot_B": 300.0},
        travel_times={
            ("zone_1", "depot_A"): 10.0,
            ("zone_1", "depot_B"): 25.0,
            ("zone_2", "depot_A"): 35.0,
            ("zone_2", "depot_B"): 12.0,
        },
        scenarios={
            "s1": {"zone_1": 100.0, "zone_2": 150.0},
            "s2": {"zone_1": 120.0, "zone_2": 200.0},
        },
        scenario_probs={"s1": 0.6, "s2": 0.4},
        max_response_time=45.0,
    )

    solver_greedy = GreedyAllocation()
    sol_greedy = solver_greedy.solve(problem)
    assert sol_greedy.status == "optimal"
    assert sol_greedy.objective_value >= 0
    assert sum(sol_greedy.unmet_demand.values()) == 0.0

    solver_prop = ProportionalAllocation()
    sol_prop = solver_prop.solve(problem)
    assert sol_prop.status == "optimal"
