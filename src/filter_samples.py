import argparse
import ast
import logging
import re
import string
from argparse import ArgumentParser
from functools import reduce
from typing import List

import nltk
from nltk.tokenize import word_tokenize
from tqdm.auto import tqdm

from utils import extract_toxic_sentences

EMOJI_PATTERN = re.compile(r"[" + re.escape(string.punctuation) + r"]", re.UNICODE)


def split_by_words(text: str, lang: str = "en") -> List[str]:
    """
    Tokenizes input text into words using nltk's word_tokenize function.

    Args:
        text (str): The input text to tokenize.
        lang (str, optional): The language of the input text. Defaults to "en".

    Returns:
        List[str]: A list of words extracted from the input text.
    """
    return word_tokenize(text, language=lang, preserve_line=False)


def cleanup(text: str) -> str:
    """
    Cleans up input text by removing unnecessary characters, URLs, and emojis.

    Args:
        text (str): The input text to clean up.

    Returns:
        str: The cleaned up text.
    """
    text = text.rstrip().lstrip()
    text = re.sub(r"\n+", "", text)
    text = re.sub(
        r"(?:((?<=[\s\W])|^)[#](\w+|[^#]|$)|((?<=[\s\W])|^)[@]([a-zA-Z0-9_]+|$))",
        "",
        text,
    )
    text = re.sub(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", "", text)
    text = EMOJI_PATTERN.sub(r"", text)
    text = re.sub(r"\s+", " ", text)
    return text


def filter_by_length(
    texts: List[str], min_length: int = 5, max_length: int = 30, lang: str = "en"
) -> List[str]:
    """
    Filters a list of sentences by their word count.

    This function takes a list of sentences and returns a new list containing only the sentences
    whose word count is within the specified range (inclusive).

    Args:
        texts (List[str]): The list of sentences to filter.
        min_length (int, optional): The minimum word count for a sentence to be included. Defaults to 5.
        max_length (int, optional): The maximum word count for a sentence to be included. Defaults to 30.

    Returns:
        List[str]: A new list containing the filtered sentences.
    """
    return [
        x
        for x in texts
        if min_length <= (word_count := len(split_by_words(x, lang=lang))) <= max_length
    ]


def main(args) -> None:
    """
    Demonstrates how to use the provided functions together.
    """

    if args.source_data_path.endswith("txt"):
        with open(args.source_data_path, "r") as f:
            texts = f.read().split("\n")

    elif args.source_data_path.endswith("csv"):
        df = pd.read_csv(args.source_data_path)
        pass
    else:
        pass

    cleaned_texts = []
    for text in tqdm(texts):
        cleaned_texts.append(cleanup(text))

    filtered_texts = filter_by_length(
        cleaned_texts, args.min_length, args.max_length, args.language
    )
    filtered_texts = list(set(filtered_texts))
    logging.info(f"Num of sentences before cleaning and dedup:{len(texts)}")
    logging.info(f"Num of sentences after cleaning and dedup:{len(filtered_texts)}")

    with open(args.save_path, "w") as f:
        f.write("\n".join(filtered_texts))


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_data_path",
        type=str,
        required=True,
        help="Path to a '.json' or `.csv` with source unfiltered data.",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path for saving processed data. File will be saved as '.txt'",
    )

    parser.add_argument(
        "--language",
        type=str,
        required=False,
        help="Language of the data.",
        default="en",
    )

    parser.add_argument(
        "--min_length",
        type=int,
        default=5,
        help="Minimum word count for a sentence to be included. Defaults to 5.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum word count for a sentence to be included. Defaults to 30.",
    )
    args = parser.parse_args()
    main(args)
