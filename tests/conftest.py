import pytest
import torch

@pytest.fixture
def sample_batch():
    B, C_sar, C_opt, H, W = 2, 2, 13, 64, 64
    return {
        'sar': torch.randn(B, C_sar, H, W),
        'optical': torch.randn(B, C_opt, H, W),
        'geo': torch.randn(B, 6, H, W),
        'rainfall': torch.randn(B, 30, 1),
        'label': torch.randint(0, 2, (B, H, W)).float(),
        'valid_mask': torch.ones(B, H, W),
        'modality_mask': torch.ones(B, 5),
    }

@pytest.fixture
def default_config():
    return {
        "model": {
            "sar_channels": 2,
            "optical_channels": 13,
            "geo_features": 6,
            "rain_features": 30,
            "hidden_dim": 64,
            "evidential": True
        }
    }
