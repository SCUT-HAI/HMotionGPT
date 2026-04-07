from pathlib import Path
from typing import Any, Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {path}")
    config["_config_path"] = str(path)
    config["_project_root"] = str(path.parent.parent.resolve())
    return config


def get_config_dir(config: Dict[str, Any]) -> Path:
    return Path(config["_config_path"]).parent


def get_project_root(config: Dict[str, Any]) -> Path:
    return Path(config["_project_root"])
