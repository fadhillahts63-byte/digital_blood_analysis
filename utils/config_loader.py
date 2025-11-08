import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml"):
    """Muat konfigurasi dari file YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)