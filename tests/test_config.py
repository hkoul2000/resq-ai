import os
import pytest
from src.utils.config import load_config

def test_load_default_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    config = load_config(config_path)
    
    assert "project" in config
    assert config["project"]["name"] == "resq-ai"
    assert "data" in config
    assert "training" in config
    assert "model" in config

def test_smoke_mode_overrides():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    config = load_config(config_path, smoke=True)
    
    assert config["project"]["smoke"] is True
    assert config["training"]["epochs"] == config["smoke"]["epochs"]
    assert config["data"]["batch_size"] == config["smoke"]["batch_size"]
