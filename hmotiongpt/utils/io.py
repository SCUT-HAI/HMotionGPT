import json
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl_line(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_path(path_str: str, base_dirs: List[Path]) -> Path:
    candidate = Path(path_str).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()
    for base_dir in base_dirs:
        resolved = (base_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    if candidate.is_absolute():
        return candidate
    return (base_dirs[0] / candidate).resolve()


def load_numpy(path: Union[str, Path]) -> np.ndarray:
    return np.load(Path(path))
