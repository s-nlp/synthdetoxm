import gc
import json
import os
import random
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

LANG_PROMPTS = {
    "es": "Write a non-toxic version of the following text in Spanish: ",
    "ru": "Write a non-toxic version of the following text in Russian: ",
    "de": "Write a non-toxic version of the following text in German: ",
    "fr": "Write a non-toxic version of the following text in French: ",
}


def tokenize_and_encode(
    example: pd.Series,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    multiparadetox: bool = False,
) -> Union[Dict[str, torch.Tensor], pd.Series]:
    """
    Tokenizes and encodes the input sentences and labels using the given tokenizer.

    Args:
        example (pd.Series): A pandas Series containing the input sentences and labels.
        tokenizer (Union[PreTrainedTokenizer, PreTrainedTokenizerFast]): The tokenizer to use for encoding.

    Returns:
        Union[Dict[str, torch.Tensor], pd.Series]: A dictionary or pandas Series with encoded input features and labels.

    """

    lang_col = 'lang' if multiparadetox else 'Lang'
    toxic_col = 'toxic_sentence' if multiparadetox else 'Toxic'
    neutral_col = 'neutral_sentence' if multiparadetox else 'Detoxed'

    prompt = LANG_PROMPTS[example[lang_col]]

    if prompt is None or example[toxic_col] is None:
        print(example)

    encoded_input = tokenizer(
        prompt + example[toxic_col],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    encoded_label = tokenizer(
        example[neutral_col],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt",
        return_attention_mask=False,
        return_token_type_ids=False,
    )

    example["input_ids"] = encoded_input["input_ids"].squeeze()
    example["attention_mask"] = encoded_input["attention_mask"].squeeze()
    example["labels"] = encoded_label["input_ids"].squeeze()

    return example


def collate_test_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
    }


def load_config(config_file_path: str) -> Dict[str, Union[str, int, float]]:
    """Load configuration from a JSON file."""
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across multiple runs.

    Args:
        seed (int): The seed value to set for random number generation.

    Returns:
        None: Only sets seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def clear_mem() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def get_batches_per_lang(
    data: pd.DataFrame,
    batch_size: int = 400,
    num_batches: int = 10,
    multiparadetox: bool = False,
):
    batches = []
    if not multiparadetox:
        data = data.sort_values(by=["STA pipe", "SIM"], ascending=[False, False])

        for i in range(num_batches):
            batch = (
                data[data.Lang != "fr"]
                .groupby("Lang")
                .apply(lambda x: x.iloc[i * batch_size : (i + 1) * batch_size])
            )
            batch = batch.reset_index(drop=True)
            batches.append(batch)
        return batches
    else:
        for i in range(num_batches):
            batch = (
                data[data.lang.isin({'ru', 'es', 'de'})]
                .groupby("lang")
                .apply(lambda x: x.iloc[i * batch_size : (i + 1) * batch_size])
            )
            batch = batch.reset_index(drop=True)
            batches.append(batch)
        return batches
 
