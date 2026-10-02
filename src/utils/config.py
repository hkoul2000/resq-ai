import os
import yaml
from typing import Dict, Any

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def apply_smoke_mode(self):
        """Override configuration with smoke settings for rapid CPU testing."""
        if not self.config.get("project", {}).get("smoke", False):
            return

        smoke_settings = self.config.get("smoke", {})
        if "epochs" in smoke_settings:
            self.config["training"]["epochs"] = smoke_settings["epochs"]
        if "batch_size" in smoke_settings:
            self.config["data"]["batch_size"] = smoke_settings["batch_size"]
        if "crop_size" in smoke_settings:
            self.config["data"]["crop_size"] = smoke_settings["crop_size"]
        if "ensemble_size" in smoke_settings:
            self.config["uq"]["ensemble_size"] = smoke_settings["ensemble_size"]
        if "mc_samples" in smoke_settings:
            self.config["uq"]["mc_samples"] = smoke_settings["mc_samples"]
        if "mc_impact_samples" in smoke_settings:
            self.config["uq"]["mc_impact_samples"] = smoke_settings["mc_impact_samples"]
        if "num_scenarios" in smoke_settings:
            self.config["optimization"]["num_scenarios"] = smoke_settings["num_scenarios"]
        if "num_workers" in smoke_settings:
            self.config["project"]["num_workers"] = smoke_settings["num_workers"]

    def get_device(self) -> str:
        """Auto-detect appropriate compute device."""
        device = self.config.get("project", {}).get("device", "auto")
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

def load_config(config_path: str, smoke: bool = False) -> Dict[str, Any]:
    loader = ConfigLoader(config_path)
    if smoke:
        if "project" not in loader.config:
            loader.config["project"] = {}
        loader.config["project"]["smoke"] = True
    
    loader.apply_smoke_mode()
    
    # Update device
    if "project" in loader.config:
        loader.config["project"]["device"] = loader.get_device()
        
    return loader.config
