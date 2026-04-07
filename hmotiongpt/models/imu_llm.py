from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


TORCH_DTYPES = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class IMULLM(nn.Module):
    def __init__(self, model_name_or_path: str, freeze_llm: bool, torch_dtype: str = "auto") -> None:
        super().__init__()
        dtype = TORCH_DTYPES[torch_dtype]
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
            )
        except TypeError:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
            )
        except Exception:
            self.llm = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        if dtype != "auto":
            self.llm = self.llm.to(dtype=dtype)
        if freeze_llm:
            for parameter in self.llm.parameters():
                parameter.requires_grad = False
        self.embed = self.llm.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        imu_embeds: torch.Tensor,
        imu_range: Dict[str, int],
    ):
        text_embeds = self.embed(input_ids)
        start = imu_range["start"]
        length = imu_range["k"]
        text_embeds[:, start : start + length, :] = imu_embeds.to(text_embeds.dtype)
        return self.llm(inputs_embeds=text_embeds, attention_mask=attention_mask, labels=labels)
