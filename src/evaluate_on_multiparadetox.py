import argparse
import logging
import os
from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import transformers
from sacrebleu import CHRF
from sentence_transformers import SentenceTransformer

from evaluate import evaluate_sim, evaluate_sta


def ensure_dir(directory: str):
    """Ensure that the directory exists, if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def evaluate_style_transfer(
    original_texts: List[str],
    rewritten_texts: List[str],
    sta_pipeline: transformers.Pipeline,
    meaning_model: SentenceTransformer,
    references: Optional[List[str]] = None,
    batch_size: int = 32,
) -> Dict[str, npt.NDArray[np.float64]]:

    accuracy = evaluate_sta(
        tox_pipe=sta_pipeline,
        texts=rewritten_texts,
    )

    similarity = evaluate_sim(
        model=meaning_model,
        original_texts=original_texts,
        rewritten_texts=rewritten_texts,
        batch_size=batch_size,
    )

    result = {
        "STA": accuracy,
        "SIM": similarity,
    }

    if references is not None:

        chrf = CHRF()

        result["CHRF"] = np.array(
            [
                chrf.sentence_score(hypothesis=rewritten, references=[reference]).score
                / 100
                for rewritten, reference in zip(rewritten_texts, references)
            ],
            dtype=np.float64,
        )

        result["J"] = result["STA"] * result["SIM"] * result["CHRF"]

    return result


def find_language_column(df: pd.DataFrame) -> str:
    for col in ["lang", "language", "Lang"]:
        if col in df.columns:
            return col
    raise ValueError("No language column found in the predictions file.")


def evaluate_by_language(
    test_data: pd.DataFrame,
    predictions: pd.DataFrame,
    sta_pipeline: transformers.Pipeline,
    batch_size: int = 32,
    output_base: str = "results/evaluation_results",
):
    lang_col = find_language_column(predictions)
    languages = predictions[lang_col].unique()
    for language in set(test_data[lang_col].unique()) - set(languages):
        test_data = test_data[test_data[lang_col] != language]

    meaning_model = SentenceTransformer("sentence-transformers/LaBSE")
    all_aggregated_results = []
    all_non_aggregated_results = []

    for lang in languages:
        logger.info(f"Evaluating for language: {lang}")

        test_subset = test_data[(predictions[lang_col] == lang).values.tolist()]
        pred_subset = predictions[predictions[lang_col] == lang]

        results = evaluate_style_transfer(
            original_texts=test_subset["toxic_sentence"].values.tolist(),
            rewritten_texts=pred_subset["neutral_sentence"].values.tolist(),
            sta_pipeline=sta_pipeline,
            meaning_model=meaning_model,
            references=test_subset["neutral_sentence"],
            batch_size=batch_size,
        )

        aggregated_results = {k: np.mean(v) for k, v in results.items()}
        logger.info(f"Aggregated Results for {lang}: {aggregated_results}")
        logger.info(results)

        all_aggregated_results.append(
            {
                "language": lang,
                "STA": aggregated_results["STA"],
                "SIM": aggregated_results["SIM"],
                "CHRF": aggregated_results["CHRF"],
                "J": aggregated_results["J"],
            }
        )

        for i, original_text in enumerate(
            test_subset["toxic_sentence"].values.tolist()
        ):
            all_non_aggregated_results.append(
                {
                    "source_text": original_text,
                    "detoxified_text": pred_subset["neutral_sentence"].values.tolist()[
                        i
                    ],
                    "reference_text": test_subset["neutral_sentence"].values.tolist()[
                        i
                    ],
                    "STA": results["STA"][i],
                    "SIM": results["SIM"][i],
                    "CHRF": results["CHRF"][i],
                    "J": results["J"][i],
                    "language": lang,
                }
            )

    aggregated_output_file = f"{output_base}_aggregated.csv"
    non_aggregated_output_file = f"{output_base}_non_aggregated.csv"

    ensure_dir(os.path.dirname(aggregated_output_file))
    ensure_dir(os.path.dirname(non_aggregated_output_file))

    aggregated_df = pd.DataFrame(all_aggregated_results)
    aggregated_df.to_csv(aggregated_output_file, index=False)
    logger.info(f"Aggregated results saved to {aggregated_output_file}")
    
    non_aggregated_df = pd.DataFrame(all_non_aggregated_results)
    non_aggregated_df.to_csv(non_aggregated_output_file, index=False)
    logger.info(f"Non-aggregated results saved to {non_aggregated_output_file}")

    logger.info(f"Aggregated results:\n\tSTA: {aggregated_df.STA.mean()}\n\tSIM: {aggregated_df.SIM.mean()}\n\tCHRF: {aggregated_df.CHRF.mean()}\n\tJ: {aggregated_df.J.mean()}")
    logger.info(f"Non-aggregated results:\n\tSTA: {non_aggregated_df.STA.mean()}\n\tSIM: {non_aggregated_df.SIM.mean()}\n\tCHRF: {non_aggregated_df.CHRF.mean()}\n\tJ: {non_aggregated_df.J.mean()}")


if __name__ == "__main__":
    """
    Basic usage:

    python evaluate_script.py \
        --test_data_path "data/predictions_for_paper/multiparadetox_test.csv" \
        --predictions_path "data/predictions_for_paper/predictions.csv" \
        --output_base "results/evaluation_results"

    will by default will save the results (both aggregated and non-aggregated into
    `results/evaluation_results_aggregated.csv` and `results/evaluation_results_non_aggregated.csv`)
    """

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data_path",
        type=str,
        required=False,
        default="data/predictions_for_paper/multiparadetox_test.csv",
        help="Path to a '.csv' file with test data.",
    )

    parser.add_argument(
        "--predictions_path",
        type=str,
        required=True,
        help="Path to `.csv` or `.tsv` with predictions.",
    )

    parser.add_argument(
        "--output_base",
        type=str,
        required=False,
        default="results/evaluation_results",
        help="Base path and filename for saving the results. Aggregated and non-aggregated files will be generated.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.test_data_path):
        logger.error(f"Test data file not found: {args.test_data_path}")
        exit(1)

    if not os.path.isfile(args.predictions_path):
        logger.error(f"Predictions file not found: {args.predictions_path}")
        exit(1)

    test_data = pd.read_csv(args.test_data_path)
    predictions = pd.read_csv(args.predictions_path)

    tox_pipe = transformers.pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier",
        device="cuda",
    )

    evaluate_by_language(
        test_data=test_data,
        predictions=predictions,
        sta_pipeline=tox_pipe,
        batch_size=32,
        output_base=args.output_base,
    )
