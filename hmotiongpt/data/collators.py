from typing import Any, Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizerBase


def build_alignment_text(example: Dict[str, Any]) -> str:
    system = "系统：你会在 <imu_start> 与 <imu_end> 之间接收 IMU 特征。"
    if "text" in example:
        user = "用户：请根据 IMU 片段生成动作描述。"
        assistant = f"助手：{example['text']}"
        return "\n".join([system, user, assistant])
    question = example.get("question", "请根据 IMU 片段回答问题。")
    answer = example.get("answer") or example.get("output") or example.get("label") or ""
    return "\n".join([system, f"用户：{question}", f"助手：{answer}"])


def build_sft_prompt_and_target(example: Dict[str, Any]) -> Tuple[str, str]:
    if "conversations" in example:
        prompt_parts = ["系统：你会在 <imu_start> 与 <imu_end> 之间接收 IMU 特征。"]
        target = ""
        for turn in example["conversations"]:
            role = "用户" if turn["from"] == "user" else "助手"
            if role == "助手":
                target = turn["value"]
            else:
                prompt_parts.append(f"{role}：{turn['value']}")
        prompt_parts.append("助手：")
        return "\n".join(prompt_parts), target
    instruction = example.get("instruction") or "Given an IMU segment, answer the question."
    model_input = example.get("input", "")
    target = example.get("output") or example.get("answer") or example.get("label") or ""
    if model_input:
        prompt = f"{instruction}\n\n{model_input}\n\nAnswer:"
    else:
        prompt = f"{instruction}\n\nAnswer:"
    return prompt, target


class AlignmentCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        imu_token_ids: Dict[str, int],
        k_tokens: int,
        max_seq_len: int,
        input_dim: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.imu_token_ids = imu_token_ids
        self.k_tokens = k_tokens
        self.max_seq_len = max_seq_len
        self.input_dim = input_dim

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        text_tensors = []
        label_tensors = []
        attn_tensors = []
        imu_tensors = []
        prefix = [self.imu_token_ids["imu_start_id"]] + [self.imu_token_ids["imu_pad_id"]] * self.k_tokens + [self.imu_token_ids["imu_end_id"]]
        for example in batch:
            text = build_alignment_text(example)
            text_ids = self.tokenizer.encode(text, add_special_tokens=True)
            ids = (prefix + text_ids)[: self.max_seq_len]
            labels = ids.copy()
            assistant_idx = text.find("助手：")
            if assistant_idx >= 0:
                prefix_text = text[:assistant_idx]
                prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=True)
                supervise_start = len(prefix) + len(prefix_ids)
            else:
                supervise_start = len(prefix)
            for idx in range(min(supervise_start, len(labels))):
                labels[idx] = -100
            for idx in range(min(len(prefix), len(labels))):
                labels[idx] = -100
            text_tensors.append(torch.tensor(ids, dtype=torch.long))
            label_tensors.append(torch.tensor(labels, dtype=torch.long))
            attn_tensors.append(torch.ones(len(ids), dtype=torch.long))
            imu_tensors.append(self._prepare_imu(torch.from_numpy(example["imu"])))
        return self._pack(text_tensors, label_tensors, attn_tensors, imu_tensors)

    def _prepare_imu(self, imu_tensor: torch.Tensor) -> torch.Tensor:
        time_steps, channels = imu_tensor.shape
        if channels < self.input_dim:
            pad = torch.zeros(time_steps, self.input_dim - channels, dtype=imu_tensor.dtype)
            return torch.cat([imu_tensor, pad], dim=1)
        if channels > self.input_dim:
            return imu_tensor[:, : self.input_dim]
        return imu_tensor

    def _pack(
        self,
        input_ids_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        attn_list: List[torch.Tensor],
        imu_list: List[torch.Tensor],
    ) -> Dict[str, Any]:
        batch_size = len(input_ids_list)
        max_len = max(t.size(0) for t in input_ids_list)
        padded_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        padded_attn = torch.zeros((batch_size, max_len), dtype=torch.long)
        for idx, tensor in enumerate(input_ids_list):
            length = tensor.size(0)
            padded_ids[idx, :length] = tensor
            padded_labels[idx, :length] = labels_list[idx]
            padded_attn[idx, :length] = attn_list[idx]
        max_time = max(t.size(0) for t in imu_list)
        imu_batch = torch.zeros((batch_size, max_time, self.input_dim), dtype=torch.float32)
        for idx, tensor in enumerate(imu_list):
            imu_batch[idx, : tensor.size(0)] = tensor
        return {
            "input_ids": padded_ids,
            "attention_mask": padded_attn,
            "labels": padded_labels,
            "imu_seq": imu_batch,
            "imu_range": {"start": 1, "k": self.k_tokens},
        }


class SFTCollator(AlignmentCollator):
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list = []
        labels_list = []
        attn_list = []
        imu_tensors = []
        prefix = [self.imu_token_ids["imu_start_id"]] + [self.imu_token_ids["imu_pad_id"]] * self.k_tokens + [self.imu_token_ids["imu_end_id"]]
        for example in batch:
            prompt, target = build_sft_prompt_and_target(example)
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            ids = (prefix + prompt_ids + target_ids)[: self.max_seq_len]
            labels = [-100] * min(len(prefix) + len(prompt_ids), len(ids))
            labels.extend(ids[len(labels) :])
            labels = labels[: len(ids)]
            attn = [1] * len(ids)
            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.long))
            attn_list.append(torch.tensor(attn, dtype=torch.long))
            imu_tensors.append(self._prepare_imu(torch.from_numpy(example["imu"])))
        batch_dict = self._pack(input_ids_list, labels_list, attn_list, imu_tensors)
        batch_dict["imu_range"] = {"start": 1, "k": self.k_tokens}
        return batch_dict
