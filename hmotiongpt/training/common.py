import math
import time
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from hmotiongpt.config import get_config_dir, get_project_root
from hmotiongpt.data.datasets import IMUDataset
from hmotiongpt.models.imu_llm import IMULLM
from hmotiongpt.models.projector import IMUProjector
from hmotiongpt.tokenizer import add_imu_tokens
from hmotiongpt.utils.io import resolve_path, save_json, save_jsonl_line
from hmotiongpt.utils.logging import create_summary_writer
from hmotiongpt.utils.seed import set_seed


def get_device(config: Dict[str, Any]) -> str:
    requested = config.get("runtime", {}).get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_model_path(config: Dict[str, Any]) -> str:
    return resolve_optional_path(config, config["model"]["name_or_path"])


def resolve_optional_path(config: Dict[str, Any], path_str: str) -> str:
    base_dirs = [get_config_dir(config), get_project_root(config), Path.cwd()]
    return str(resolve_path(path_str, base_dirs))


def build_dataset(config: Dict[str, Any]) -> IMUDataset:
    data_cfg = config["data"]
    jsonl_path = resolve_optional_path(config, data_cfg["jsonl"])
    imu_roots = [resolve_optional_path(config, root) for root in data_cfg.get("imu_roots", [])]
    return IMUDataset(jsonl_path=jsonl_path, imu_roots=imu_roots)


def create_tokenizer_and_model(config: Dict[str, Any], freeze_llm: bool):
    model_path = resolve_model_path(config)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = IMULLM(
        model_name_or_path=model_path,
        freeze_llm=freeze_llm,
        torch_dtype=config.get("runtime", {}).get("torch_dtype", "auto"),
    )
    imu_token_ids = add_imu_tokens(tokenizer, model.llm)
    return tokenizer, model, imu_token_ids


def create_projector(config: Dict[str, Any], d_model: int) -> IMUProjector:
    projector_cfg = config["projector"]
    return IMUProjector(
        d_model=d_model,
        k_tokens=projector_cfg["num_tokens"],
        input_dim=projector_cfg["input_dim"],
        hidden_dim=projector_cfg["hidden_dim"],
        mode=projector_cfg.get("mode", "pool"),
    )


def create_scheduler(optimizer, num_epochs: int, steps_per_epoch: int, warmup_steps: int):
    total_steps = max(1, num_epochs * steps_per_epoch)
    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )


def create_run_dir(config: Dict[str, Any]) -> Path:
    output_root = Path(resolve_optional_path(config, config["output"]["root"]))
    run_name = config["output"].get("run_name") or time.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_writer(run_dir: Path):
    return create_summary_writer(run_dir / "tensorboard")


def save_run_config(run_dir: Path, config: Dict[str, Any]) -> None:
    payload = {key: value for key, value in config.items() if not key.startswith("_")}
    save_json(run_dir / "config.json", payload)


def save_epoch_metrics(run_dir: Path, payload: Dict[str, Any]) -> None:
    save_jsonl_line(run_dir / "metrics.jsonl", payload)


def safe_ppl(loss_value: float) -> float:
    return math.exp(loss_value) if loss_value < 20 else 1e4


def prepare_runtime(config: Dict[str, Any]) -> str:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    return get_device(config)
