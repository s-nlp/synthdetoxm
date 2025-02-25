import argparse
import pathlib
import logging

import numpy as np
import pandas as pd
import transformers
from sentence_transformers import SentenceTransformer

from evaluate import evaluate_sim, evaluate_sta

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
    )

    original = pathlib.Path(args.toxic_data_path).read_text().splitlines()
    detoxed_path = pathlib.Path(args.detox_data_path)

    logging.info(len(original))
    df = pd.DataFrame(
        {
            "toxic_text": original,
            "sta_toxic": (
                1 - evaluate_sta(tox_pipe, original, method="pipeline")
            ).tolist(),
        }
    )

    for file in detoxed_path.iterdir():

        file_name = file.name

        if file_name.endswith(".csv"):
            continue

        rewritten = file.read_text().splitlines()

        temp_df = pd.DataFrame(
            {
                f"detoxed_{file_name}": rewritten,
                f"sta_detoxed_{file_name}": (
                    1 - evaluate_sta(tox_pipe, rewritten, method="pipeline")
                ).tolist(),
                f"sim_{file_name}": evaluate_sim(model, original, rewritten).tolist(),
            }
        )

        df = pd.concat([df, temp_df], axis=1)

    df.reset_index(drop=True, inplace=True)
    df.to_csv(detoxed_path / pathlib.Path("concat_new.csv"), index=False)

    for col in df.columns:
        if col.startswith("sta"):
            logging.info(f"{col}: {np.mean(df[col])}")

    for col in df.columns:
        if col.startswith("sim"):
            logging.info(f"{col}: {np.mean(df[col])}")
