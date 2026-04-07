from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

from hmotiongpt.utils.io import load_jsonl, resolve_path


class IMUDataset(Dataset):
    def __init__(self, jsonl_path: str, imu_roots: Optional[List[str]] = None) -> None:
        self.jsonl_path = Path(jsonl_path).expanduser().resolve()
        self.items = load_jsonl(self.jsonl_path)
        self.search_roots = [self.jsonl_path.parent]
        for root in imu_roots or []:
            self.search_roots.append(resolve_path(root, [self.jsonl_path.parent, Path.cwd()]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = dict(self.items[index])
        imu_field = (
            example.get("imu_vec_path")
            or example.get("imu_path")
            or example.get("imu_file")
            or example.get("imu")
        )
        if imu_field is None:
            raise ValueError("Missing IMU path field in sample")
        imu_path = resolve_path(imu_field, self.search_roots)
        imu_array = np.load(imu_path).astype("float32")
        example["imu"] = imu_array
        example["imu_resolved_path"] = str(imu_path)
        return example
