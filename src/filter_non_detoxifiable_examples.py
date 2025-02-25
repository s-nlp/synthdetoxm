import argparse
import pathlib
import logging

import numpy as np
import pandas as pd
import transformers
from sentence_transformers import SentenceTransformer

from evaluate import evaluate_sim, evaluate_sta

def classify_refusals(refusal_pipe, sentences, threshold=0.6):
    """
    Classifies sentences as refusals or not using the refusal classification pipeline.

    Args:
        refusal_pipe: Refusal classification pipeline.
        sentences: List of sentences to classify.
        threshold: Probability threshold for classifying as a refusal.

    Returns:
        A list of booleans indicating whether each sentence is a refusal.
    """
    classifications = []
    for sentence in sentences:
        result = refusal_pipe(sentence)[0]
        is_refusal = result["label"] == "REFUSAL" and result["score"] > threshold
        classifications.append(is_refusal)
    return classifications


def calculate_mean_detox_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the mean absolute reduction in toxicity across all files,
    skipping files marked as refusals. Ensures reduction is always positive.

    Args:
        df: The concatenated DataFrame containing both detoxed and toxic data.

    Returns:
        The DataFrame with a new column 'mean_absolute_reduction' containing the calculated mean absolute reduction (always positive).
    """

    def calculate_absolute_reduction(toxic_score, detoxed_score):
        """Calculates absolute reduction (toxic - detoxed) and takes its absolute value."""
        if isinstance(toxic_score, float) and isinstance(detoxed_score, float):
            return abs(toxic_score - detoxed_score)
        else:
            print(toxic_score, type(toxic_score), detoxed_score, type(detoxed_score))
            return 0.000001

    def calculate_row_metric(row, t):
        """Calculates the mean absolute reduction for a row."""
        absolute_reductions = []
        toxic_score = row[f"sta_toxic_{t}"]
        for col in row.index:
            if col.startswith(f"sta_detoxed_{t}_"):
                file_name = col.replace(f"sta_detoxed_{t}_", "")
                refusal_col = f"is_refusal_{file_name}"
                if refusal_col in row.index and row[refusal_col]:
                    continue  # Skip if it's a refusal
                detoxed_score = row[col]
                absolute_reduction = calculate_absolute_reduction(toxic_score, detoxed_score)
                try:
                    absolute_reductions.append(absolute_reduction / toxic_score if toxic_score != 0 else 0)
                except:
                    print(row)
                    raise

        if not absolute_reductions:
            return np.nan
        else:
            return np.mean(absolute_reductions)

    mean_absolute_reduction_pipe = df.apply(lambda x: calculate_row_metric(x, 'pipe'), axis=1)
    mean_absolute_reduction_api = df.apply(lambda x: calculate_row_metric(x, 'api'), axis=1)

    df['mean_absolute_reduction_pipe'] = mean_absolute_reduction_pipe
    df['mean_absolute_reduction_api'] = mean_absolute_reduction_api

    return df


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--toxic_data_path",
        type=str,
        required=True,
        help="Path to toxic data.",
    )
    parser.add_argument(
        "--detox_data_path",
        type=str,
        required=True,
        help="Path to detoxed data.",
    )

    args = parser.parse_args()

    model = SentenceTransformer("sentence-transformers/LaBSE")
    tox_pipe = transformers.pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier",
        device="cuda",
        truncation=True
    )
    refusal_pipe = transformers.pipeline(
        "text-classification",
        model="chameleon-lizard/xlmr-base-refusal-classifier",
        device="cuda",
        truncation=True
    )

    original = pathlib.Path(args.toxic_data_path).read_text().splitlines()
    detoxed_path = pathlib.Path(args.detox_data_path)
    lang = args.detox_data_path.split('/')[-1]

    logging.info(len(original))

    if lang == 'fr':
        stas_pipe = (1 - evaluate_sta(tox_pipe, original, method='api')).tolist()
        stas_api = stas_pipe
    else:
        stas_pipe = (1 - evaluate_sta(tox_pipe, original, method='pipeline')).tolist()
        stas_api = (1 - evaluate_sta(tox_pipe, original, method='api')).tolist()
    df = pd.read_csv(detoxed_path / pathlib.Path('concat_new_detoxifiability.csv'))

    for file in detoxed_path.iterdir():
        file_name = file.name

        if not file_name.endswith("_classified.tsv"):
            continue

        file = pd.read_csv(file, sep='\t')
        try:
            rewritten = file['detox_sentence'].tolist()
            refusals = file['overall_refusal'].tolist()
        except KeyError:
            print(file.columns)
            raise

        sims = evaluate_sim(model, original, rewritten)
        df[f'sim_detoxed_{file_name}'] = sims

        try:
            if lang == 'fr':
                stas_pipe = (1 - evaluate_sta(tox_pipe, rewritten, method='api')).tolist()
                stas_api = stas_pipe
            else:
                stas_pipe = (1 - evaluate_sta(tox_pipe, rewritten, method='pipeline')).tolist()
                stas_api = (1 - evaluate_sta(tox_pipe, rewritten, method='api')).tolist()
        except ValueError:
            print(rewritten[:5], file_name)
            raise

        temp_df = pd.DataFrame(
            {
                f"detoxed_{file_name}": rewritten,
                f"sta_detoxed_pipe_{file_name}": stas_pipe,
                f"sta_detoxed_api_{file_name}": stas_api,
                f'sim_detoxed_{file_name}': sims,
                f"is_refusal_{file_name}": refusals,
            }
        )

        df = pd.concat([df, temp_df], axis=1)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(detoxed_path / pathlib.Path("concat_new_final.csv"), index=False)

    df = pd.read_csv(detoxed_path / pathlib.Path('concat_new.csv'))

# Calculate the mean absolute reduction
    final_df = calculate_mean_detox_delta(df)

# Classify refusals based on a threshold
    refusal_threshold = 0.5  # Set threshold for absolute reduction
    final_df['detoxifiable_api'] = final_df['mean_absolute_reduction_api'] >= refusal_threshold
    final_df['detoxifiable_pipe'] = final_df['mean_absolute_reduction_pipe'] >= refusal_threshold

    final_df.reset_index(drop=True, inplace=True)
    final_df.to_csv(detoxed_path / pathlib.Path("concat_new_detoxifiability.csv"), index=False)

    # Print classification stats
    api_detoxifiable_count = final_df['detoxifiable_api'].sum()
    pipe_detoxifiable_count = final_df['detoxifiable_pipe'].sum()
    total_count = len(final_df)
    logging.info(f"Total sentences: {total_count}")
    logging.info(f"Number of sentences with 'detoxifiable' == True: {api_detoxifiable_count} in API")
    logging.info(f"Number of sentences with 'detoxifiable' == True: {pipe_detoxifiable_count} in pipe")

