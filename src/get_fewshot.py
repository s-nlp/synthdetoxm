"""Usage: python get_fewshot de

"""

import argparse

import datasets
import scipy
import torch
import transformers
from sentence_transformers import SentenceTransformer

labse = SentenceTransformer("sentence-transformers/LaBSE", device="cuda")
tox_pipe = transformers.pipeline(
    "text-classification",
    model="textdetox/xlmr-large-toxicity-classifier",
    device="cuda",
)


def sta(toxic_sentence: str, neutral_sentence: str) -> float:
    tox_pred = torch.softmax(
        tox_pipe.model(
            **tox_pipe.tokenizer(
                toxic_sentence,
                return_tensors="pt",
            ).to("cuda")
        ).logits,
        dim=-1,
    )[0][0]

    detox_pred = torch.softmax(
        tox_pipe.model(
            **tox_pipe.tokenizer(
                neutral_sentence,
                return_tensors="pt",
            ).to("cuda")
        ).logits,
        dim=-1,
    )[0][0]

    return tox_pred / detox_pred


def sim(toxic_sentence: str, neutral_sentence: str) -> float:
    return 1 - scipy.spatial.distance.cosine(
        labse.encode(toxic_sentence), labse.encode(neutral_sentence)
    )


def score_pair(toxic_sentence: str, neutral_sentence: str) -> float:
    return 1 - sta(toxic_sentence, neutral_sentence) * sim(
        toxic_sentence, neutral_sentence
    )


def main():
    parser = argparse.ArgumentParser(description="Get language code and result path")

    parser.add_argument(
        "language",
        choices=["de", "ru", "es", "en"],
        help="Language code (de, ru, en or es)",
    )

    args = parser.parse_args()

    print(f"Language code: {args.language}")

    data = datasets.load_dataset("textdetox/multilingual_paradetox")

    mapped_dataset = data[args.language].map(
        lambda examples: {"score": score_pair(**examples)}
    )
    sorted_dataset = mapped_dataset.sort("score", reverse=True)

    ten_shot = sorted_dataset.select(range(10))

    print(ten_shot.to_json(f"few_shot_{args.language}.json"))

    for line in ten_shot:
        print(
            "toxic:\n",
            line["toxic_sentence"],
            "\nneutral\n",
            line["neutral_sentence"],
            "\nscore\n",
            line["score"],
        )


if __name__ == "__main__":
    main()
