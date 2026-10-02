"""Unit tests for ResQNet model components.

Tests all model building blocks and the complete forward/backward pass.
All tests run on CPU without data downloads.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFiLMLayer:
    """Tests for FiLM conditioning layer."""

    def test_4d_input(self):
        from src.models.resqnet import FiLMLayer
        film = FiLMLayer(feature_dim=64, cond_dim=32)
        x = torch.randn(2, 64, 8, 8)
        cond = torch.randn(2, 32)
        out = film(x, cond)
        assert out.shape == x.shape

    def test_2d_input(self):
        from src.models.resqnet import FiLMLayer
        film = FiLMLayer(feature_dim=64, cond_dim=32)
        x = torch.randn(2, 64)
        cond = torch.randn(2, 32)
        out = film(x, cond)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        from src.models.resqnet import FiLMLayer
        film = FiLMLayer(feature_dim=64, cond_dim=32)
        x = torch.randn(2, 64, 4, 4, requires_grad=True)
        cond = torch.randn(2, 32, requires_grad=True)
        out = film(x, cond)
        out.sum().backward()
        assert x.grad is not None
        assert cond.grad is not None


class TestGatedCrossAttention:
    """Tests for gated cross-attention module."""

    def test_output_shape(self):
        from src.models.resqnet import GatedCrossAttention
        attn = GatedCrossAttention(dim=64, num_heads=4)
        q = torch.randn(2, 16, 64)
        kv = torch.randn(2, 8, 64)
        out = attn(q, kv)
        assert out.shape == q.shape

    def test_single_kv(self):
        from src.models.resqnet import GatedCrossAttention
        attn = GatedCrossAttention(dim=64, num_heads=4)
        q = torch.randn(2, 16, 64)
        kv = torch.randn(2, 1, 64)
        out = attn(q, kv)
        assert out.shape == q.shape


class TestGeoEncoder:
    """Tests for geographic feature encoder."""

    def test_output_shape(self):
        from src.models.resqnet import GeoEncoder
        enc = GeoEncoder(in_channels=6, out_dim=128)
        x = torch.randn(2, 6, 64, 64)
        out = enc(x)
        assert out.shape == (2, 128)

    def test_different_input_sizes(self):
        from src.models.resqnet import GeoEncoder
        enc = GeoEncoder(in_channels=6, out_dim=128)
        for size in [32, 64, 128]:
            x = torch.randn(1, 6, size, size)
            out = enc(x)
            assert out.shape == (1, 128)


class TestRainfallEncoder:
    """Tests for rainfall temporal encoder."""

    def test_output_shape(self):
        from src.models.resqnet import RainfallEncoder
        enc = RainfallEncoder(
            seq_len=30, d_model=32, nhead=2, num_layers=1, out_dim=64
        )
        x = torch.randn(2, 30, 1)
        out = enc(x)
        assert out.shape == (2, 64)

    def test_2d_input(self):
        """Test with 2D input (no channel dim)."""
        from src.models.resqnet import RainfallEncoder
        enc = RainfallEncoder(
            seq_len=30, d_model=32, nhead=2, num_layers=1, out_dim=64
        )
        x = torch.randn(2, 30)
        out = enc(x)
        assert out.shape == (2, 64)


class TestResQNet:
    """Tests for the complete ResQNet model."""

    @pytest.fixture
    def model(self):
        from src.models.resqnet import ResQNet
        return ResQNet(
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

    def test_output_shape(self, model, sample_batch):
        model.eval()
        with torch.no_grad():
            out = model(
                sar=sample_batch["sar"],
                optical=sample_batch["optical"],
                geo=sample_batch["geo"],
                rainfall=sample_batch["rainfall"],
                modality_mask=sample_batch["modality_mask"],
            )
        assert "probs" in out
        assert "logits" in out
        assert out["probs"].shape[0] == 2  # batch size

    def test_probability_range(self, model, sample_batch):
        model.eval()
        with torch.no_grad():
            out = model(
                sar=sample_batch["sar"],
                optical=sample_batch["optical"],
                geo=sample_batch["geo"],
                rainfall=sample_batch["rainfall"],
                modality_mask=sample_batch["modality_mask"],
            )
        assert out["probs"].min() >= 0
        assert out["probs"].max() <= 1

    def test_sar_only(self, model, sample_batch):
        """Test with only SAR modality."""
        model.eval()
        mask = torch.zeros(2, 5)
        mask[:, 0] = 1  # Only SAR available
        with torch.no_grad():
            out = model(
                sar=sample_batch["sar"],
                optical=torch.zeros_like(sample_batch["optical"]),
                geo=torch.zeros_like(sample_batch["geo"]),
                rainfall=torch.zeros_like(sample_batch["rainfall"]),
                modality_mask=mask,
            )
        assert "probs" in out

    def test_optical_only(self, model, sample_batch):
        """Test with only optical modality."""
        model.eval()
        mask = torch.zeros(2, 5)
        mask[:, 1] = 1  # Only optical available
        with torch.no_grad():
            out = model(
                sar=torch.zeros_like(sample_batch["sar"]),
                optical=sample_batch["optical"],
                geo=torch.zeros_like(sample_batch["geo"]),
                rainfall=torch.zeros_like(sample_batch["rainfall"]),
                modality_mask=mask,
            )
        assert "probs" in out

    def test_evidential_mode(self, sample_batch):
        """Test evidential output mode."""
        from src.models.resqnet import ResQNet
        model = ResQNet(
            sar_channels=2,
            optical_channels=13,
            geo_channels=6,
            pretrained=False,
            rainfall_d_model=32,
            rainfall_nhead=2,
            rainfall_layers=1,
            evidential=True,
        )
        model.eval()
        with torch.no_grad():
            out = model(
                sar=sample_batch["sar"],
                optical=sample_batch["optical"],
                geo=sample_batch["geo"],
                rainfall=sample_batch["rainfall"],
                modality_mask=sample_batch["modality_mask"],
            )
        assert "gamma" in out
        assert "nu" in out
        assert "alpha" in out
        assert "beta" in out
        assert "aleatoric_uncertainty" in out
        assert "epistemic_uncertainty" in out

    def test_modality_dropout(self, sample_batch):
        """Test that modality dropout works during training."""
        from src.models.resqnet import ResQNet
        model = ResQNet(
            sar_channels=2,
            optical_channels=13,
            geo_channels=6,
            pretrained=False,
            modality_dropout=0.5,
            rainfall_d_model=32,
            rainfall_nhead=2,
            rainfall_layers=1,
        )
        model.train()
        # Should not crash even with dropout
        out = model(
            sar=sample_batch["sar"],
            optical=sample_batch["optical"],
            geo=sample_batch["geo"],
            rainfall=sample_batch["rainfall"],
            modality_mask=sample_batch["modality_mask"],
        )
        assert "probs" in out


class TestMCDropout:
    """Tests for MC Dropout prediction."""

    def test_mc_dropout(self, sample_batch):
        from src.models.resqnet import ResQNet, mc_dropout_predict
        model = ResQNet(
            sar_channels=2, optical_channels=13, geo_channels=6,
            pretrained=False, dropout=0.3,
            rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1,
        )
        result = mc_dropout_predict(
            model, num_samples=3,
            sar=sample_batch["sar"],
            optical=sample_batch["optical"],
            geo=sample_batch["geo"],
            rainfall=sample_batch["rainfall"],
            modality_mask=sample_batch["modality_mask"],
        )
        assert "probs" in result
        assert "std" in result
        assert "epistemic_uncertainty" in result
        assert "aleatoric_uncertainty" in result
        assert result["all_probs"].shape[0] == 3  # num_samples


class TestTTA:
    """Tests for Test-Time Augmentation."""

    def test_tta_predict(self, sample_batch):
        from src.models.resqnet import ResQNet, tta_predict
        model = ResQNet(
            sar_channels=2, optical_channels=13, geo_channels=6,
            pretrained=False,
            rainfall_d_model=32, rainfall_nhead=2, rainfall_layers=1,
        )
        result = tta_predict(
            model,
            sar=sample_batch["sar"],
            optical=sample_batch["optical"],
            geo=sample_batch["geo"],
            rainfall=sample_batch["rainfall"],
            modality_mask=sample_batch["modality_mask"],
        )
        assert "probs" in result
        assert "std" in result
        assert result["all_probs"].shape[0] == 8  # 8 TTA transforms
