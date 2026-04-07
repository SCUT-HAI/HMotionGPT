from pathlib import Path
from typing import Union


class NullWriter:
    def add_scalar(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def create_summary_writer(log_dir: Union[str, Path]):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        return NullWriter()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))
