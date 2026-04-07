import torch
import torch.nn as nn


class IMUProjector(nn.Module):
    def __init__(
        self,
        d_model: int,
        k_tokens: int,
        input_dim: int,
        hidden_dim: int,
        mode: str = "pool",
    ) -> None:
        super().__init__()
        self.k_tokens = k_tokens
        self.mode = mode
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(0.5))

    def _time_align(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels = x.shape
        if self.mode == "sample":
            index = torch.linspace(0, max(time_steps - 1, 1), self.k_tokens, device=x.device)
            index = index.round().long().clamp(0, time_steps - 1)
            return x.index_select(dim=1, index=index)
        segment_ids = torch.clamp(
            ((torch.arange(time_steps, device=x.device).float() + 0.5) / time_steps * self.k_tokens).floor().long(),
            0,
            self.k_tokens - 1,
        )
        output = torch.zeros(batch_size, self.k_tokens, channels, device=x.device, dtype=x.dtype)
        counts = torch.zeros(batch_size, self.k_tokens, 1, device=x.device, dtype=x.dtype)
        expanded_ids = segment_ids.view(1, time_steps, 1).expand(batch_size, time_steps, channels)
        output.scatter_add_(1, expanded_ids, x)
        count_ids = segment_ids.view(1, time_steps, 1).expand(batch_size, time_steps, 1)
        counts.scatter_add_(1, count_ids, torch.ones(batch_size, time_steps, 1, device=x.device, dtype=x.dtype))
        return output / counts.clamp_min(1.0)

    def forward(self, imu_seq: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(imu_seq)
        aligned = self._time_align(projected)
        return torch.tanh(self.gate) * aligned
