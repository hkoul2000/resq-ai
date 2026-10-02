"""Baseline models for comparison."""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Any, Dict

class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, classes: int = 1, encoder_name: str = 'resnet34', encoder_weights: str = 'imagenet'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class EarlyFusionUNet(nn.Module):
    def __init__(self, sar_channels: int = 2, opt_channels: int = 3, classes: int = 1, encoder_name: str = 'resnet34'):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=sar_channels + opt_channels,
            classes=classes,
        )

    def forward(self, sar: torch.Tensor, opt: torch.Tensor) -> torch.Tensor:
        x = torch.cat([sar, opt], dim=1)
        return self.model(x)

class TabularBaseline:
    def __init__(self, model_type: str = 'rf', **kwargs):
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(**kwargs)
        elif model_type == 'xgb':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(**kwargs)
            except ImportError:
                raise ImportError("xgboost is not installed")
        else:
            raise ValueError(f"Unknown tabular model type: {model_type}")

    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def predict(self, X):
        return self.model.predict(X)

def create_baseline(name: str, **kwargs) -> Any:
    models = {
        'unet': UNetBaseline,
        'early_fusion': EarlyFusionUNet,
        'tabular': TabularBaseline,
    }
    if name not in models:
        raise ValueError(f"Baseline {name} not found.")
    return models[name](**kwargs)
