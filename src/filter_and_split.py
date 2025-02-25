import logging
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from utils import get_sta_scores


def process_text_entry(text: str, lang: str) -> Dict:
    """
    A helper function to apply get_sta_scores to a single row of the dataset.

    Args:
        text (str): The text content to be analyzed.
        lang (str): The language of the text.

    Returns:
        dict: The result from the Perspective API, including toxic sentences.
    """
    result = get_sta_scores(text, lang=lang)
    return result


def process_dataset_in_parallel(data: pd.DataFrame, max_workers: int = 4) -> List[Dict]:
    """
    Processes a dataset in parallel using ThreadPoolExecutor.

    Args:
        data (pd.DataFrame): The DataFrame containing text and language information.
        max_workers (int): The number of threads to use for parallel processing.

    Returns:
        List[Dict]: A list of results for each row in the dataset.
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in data.iterrows():
            future = executor.submit(process_text_entry, row["content"], row["lang"])
            futures.append(future)

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing texts"
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing text: {e}")
                results.append({"error": str(e)})

    return results


if __name__ == "__main__":

    parser = ArgumentParser("Process combined dataset with Perspective API.")
    parser.add_argument(
        "--source_data_path",
        type=str,
        required=True,
        help="Path to the source toxic multilingual data.",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        required=False,
        default=5,
        help="Number of workers for multiprocessing. ",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="data/",
        help="Path where to save processsed data set. Default is `data/`.",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        required=False,
        default="processed.csv",
        help="Name of the processed data set. Default is `processed_multilingual.csv`.",
    )

    args = parser.parse_args()

    old_data = pd.read_json("data/processed.json")
    new_data = pd.read_csv(args.source_data_path, sep="\t", index_col=0)

    old_texts = set(old_data.text.values)

    new_lines = new_data[~new_data.content.isin(old_texts)]

    results = process_dataset_in_parallel(new_lines, max_workers=1)
    result_df = pd.DataFrame(results)

    result_df = pd.concat([result_df, old_data], ignore_index=True)
    # result_df.to_csv("res.csv", index=False)

    result_df.to_csv(f"{args.output_path}/{args.output_filename}", index=False)
