import argparse
import logging
import pathlib
from typing import List

import numpy as np
import numpy.typing as npt
import torch
import transformers
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
from tqdm.auto import trange

from api import get_sta_scores


def evaluate_sim(
    model: SentenceTransformer,
    original_texts: List[str],
    rewritten_texts: List[str],
    batch_size: int = 32,
    efficient_version: bool = False,
) -> npt.NDArray[np.float64]:
    """
    Evaluate the semantic similarity between original and rewritten texts.

    Args:
        model (SentenceTransformer): The sentence transformer model.
        original_texts (List[str]): List of original texts.
        rewritten_texts (List[str]): List of rewritten texts.
        batch_size (int): Batch size for inference.
        efficient_version (bool): To use efficient calculation method.

    Returns:
        npt.NDArray[np.float64]: Array of semantic similarity scores between \
              original and rewritten texts.
    """
    similarities = []

    batch_size = min(batch_size, len(original_texts))
    for i in trange(0, len(original_texts), batch_size, desc="Calculating SIM scores"):
        original_batch = original_texts[i : i + batch_size]
        rewritten_batch = rewritten_texts[i : i + batch_size]

        embeddings = model.encode(original_batch + rewritten_batch)
        original_embeddings = embeddings[: len(original_batch)]
        rewritten_embeddings = embeddings[len(original_batch) :]

        if efficient_version:
            similarity_matrix = np.dot(original_embeddings, rewritten_embeddings.T)
            original_norms = np.linalg.norm(original_embeddings, axis=1)
            rewritten_norms = np.linalg.norm(rewritten_embeddings, axis=1)
            similarity_matrix = 1 - similarity_matrix / (
                np.outer(original_norms, rewritten_norms) + 1e-9
            )
            similarities.extend(similarity_matrix.diagonal())
        else:
            t = [
                1 - cosine(original_embedding, rewritten_embedding)
                for original_embedding, rewritten_embedding in zip(
                    original_embeddings, rewritten_embeddings
                )
            ]
            similarities.extend(t)

    return np.array(similarities, dtype=np.float64)


def calculate_toxicities(
    tox_pipe: transformers.Pipeline,
    batch: List[str],
    method: str = "pipeline",
    lang: str = "ru",
):
    try:
        if method == "pipeline":
            return [
                torch.softmax(
                    tox_pipe.model(
                        **tox_pipe.tokenizer(
                            _,
                            return_tensors="pt",
                            max_length=512,
                            truncation=True,
                        ).to("cuda")
                    ).logits[0],
                    dim=-1,
                )
                .to("cpu")
                .detach()[0]
                if _ is not np.nan else 1.0
                for _ in batch
            ]
        elif method == "api":
            res = []
            for sentence in batch:
                toxicity_score = -5
                while toxicity_score <= 0:
                    try:
                        resp = get_sta_scores(
                            sentence,
                            return_spans=False,
                            return_full_response=True,
                            lang=lang,
                        )
                        toxicity_score = resp["response"]["attributeScores"]["TOXICITY"][
                            "summaryScore"
                        ]["value"]
                        break
                    except KeyError:
                        print(resp)
                        toxicity_score += 1
                else:
                    toxicity_score = 1
                res.append(1 - toxicity_score)

        return res
    except ValueError:
        print(batch, type(batch), all(map(lambda _: isinstance(_, str), batch)))
        raise


def evaluate_sta(
    tox_pipe: transformers.pipeline,
    texts: List[str],
    batch_size: int = 32,
    method: str = "pipeline",
    lang: str = "ru",
) -> npt.NDArray[np.float64]:
    toxicities = []

    batch_size = min(batch_size, len(texts))
    for i in trange(0, len(texts), batch_size, desc="Calculating STA scores"):
        batch = texts[i : i + batch_size]

        t = calculate_toxicities(tox_pipe, batch, method=method, lang=lang)
        toxicities.extend(t)

    return np.array(toxicities, dtype=np.float64)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = SentenceTransformer("sentence-transformers/LaBSE")
    tox_pipe = transformers.pipeline(
        "text-classification",
        model="textdetox/xlmr-large-toxicity-classifier",
        device="cuda",
    )

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

    original = pathlib.Path(args.toxic_data_path).read_text().splitlines()
    rewritten = pathlib.Path(args.detox_data_path).read_text().splitlines()

    logging.info(
        f"STA: {np.mean(evaluate_sta(tox_pipe, rewritten, method='pipeline'))}"
    )
    logging.info(f"SIM: {np.mean(evaluate_sim(model, original, rewritten))}")
