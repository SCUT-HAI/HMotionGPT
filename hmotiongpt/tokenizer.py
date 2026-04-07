from transformers import PreTrainedModel, PreTrainedTokenizerBase
from typing import Dict


SPECIAL_TOKENS = ["<imu_start>", "<imu_end>", "<imu_pad>"]


def add_imu_tokens(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> Dict[str, int]:
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))
    imu_start_id, imu_end_id, imu_pad_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    return {
        "imu_start_id": imu_start_id,
        "imu_end_id": imu_end_id,
        "imu_pad_id": imu_pad_id,
    }
