import logging
import sys
from typing import Optional

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup structured logging with console and optional file output."""
    logger = logging.getLogger("resq-ai")
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def setup_mlflow(config: dict):
    """Helper to initialize MLflow tracking."""
    try:
        import mlflow
        tracking_uri = config.get("mlflow", {}).get("tracking_uri", "mlruns")
        experiment_name = config.get("mlflow", {}).get("experiment_name", "resq-ai")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    except ImportError:
        pass
