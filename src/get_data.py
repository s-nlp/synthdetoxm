import pathlib
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset
from fast_langdetect import detect


def jigsaw() -> pd.DataFrame:
    df_data = pd.read_csv("data/jigsaw_test.csv")
    df_labels = pd.read_csv("data/jigsaw_test_labels.csv")

    df = pd.merge(df_data, df_labels, on="id", how="left")
    df = df[
        (df.toxic == 1) & (df.lang.isin(["ru", "fr", "es"]))
    ]  # {'ru': 2636, 'fr': 3340, 'es': 3358}
    df = df.drop("id", axis=1)
    df = df[["content", "toxic", "lang"]]

    return df


def russian_language_toxic_comments() -> pd.DataFrame:
    df = pd.read_csv("data/russian_language_toxic_comments.csv")

    df = df.loc[df.toxic == 1.0].copy()
    df = df.rename(columns={"comment": "content"})
    df.toxic = df.toxic.astype(int)
    df["lang"] = "ru"

    return df


def toxic_russian_comments() -> pd.DataFrame:
    data = pathlib.Path("data/toxic_russian_comments.txt").read_text()
    # code adapted from
    # kaggle.com/code/alexandersemiletov/starter-read-toxic-russian-comments-dataset
    data_list = []
    for line in data.splitlines():
        labels = line.split()[0]
        text = line[len(labels) + 1 :].strip()
        labels = labels.split(",")
        toxic = int(
            (1 if "__label__INSULT" in labels else 0)
            + (1 if "__label__THREAT" in labels else 0)
            + (1 if "__label__OBSCENITY" in labels else 0)
            > 1
        )
        data_list.append((text, toxic))

    df = pd.DataFrame(data_list, columns=["content", "toxic"])
    df = df[df.toxic == 1]
    df["lang"] = "ru"

    return df


def clandestino() -> pd.DataFrame:
    # https://github.com/microsoft/Clandestino
    df = pd.read_json("data/clandestino.json")
    df["toxic"] = df.Annotators.apply(
        lambda x: np.mean(
            [
                _["Question 1: In your opinion, will this text be harmful to anyone?"]
                for _ in x
            ]
        )
    )
    df = df[df.toxic > 2.2]
    df["toxic"] = 1.0
    df.toxic.astype(int)
    df = df.rename(columns={"Sentence": "content"})
    df = df.drop(["Key", "IsAIGenerated", "Index", "Switch", "Annotators"], axis=1)
    df["lang"] = "es"
    df = df[["content", "toxic", "lang"]]

    return df


def germeval() -> pd.DataFrame:
    #
    df_train = pd.read_csv("data/GermEval21_TrainData.csv")
    df_test = pd.read_csv("data/GermEval21_TestData.csv")
    df = pd.concat([df_train, df_test], ignore_index=True)

    df = df.rename(columns={"Sub1_Toxic": "toxic", "comment_text": "content"})
    df = df[df.toxic == 1]
    df = df.drop(["comment_id", "Sub2_Engaging", "Sub3_FactClaiming"], axis=1)
    df["lang"] = "de"
    df = df[["content", "toxic", "lang"]]

    return df


def rp() -> pd.DataFrame:
    df = pd.read_csv("data/rp.csv")
    df.fillna(0)
    df = df[(df["Insult Count Crowd"] > 0) & (df["Profanity Count Crowd"] > 0)]
    df = df.drop(
        [
            "Unnamed: 0.1",
            "Unnamed: 0",
            "id",
            "Reject Newspaper",
            "Reject Crowd",
            "Rejection Count Crowd",
            "Sexism Count Crowd",
            "Racism Count Crowd",
            "Threat Count Crowd",
            "Insult Count Crowd",
            "Profanity Count Crowd",
            "Meta Count Crowd",
            "Advertisement Count Crowd",
        ],
        axis=1,
    )
    df = df.rename(columns={"Text": "content"})
    df["toxic"] = 1
    df["lang"] = "de"

    return df


def mlma() -> pd.DataFrame:
    ds = load_dataset("nedjmaou/MLMA_hate_speech")
    sents = []
    for item in ds["train"]:
        if item["sentiment"] != "normal" and item["directness"] == "direct":
            lang = detect(item["tweet"])["lang"]
            if lang == "fr":
                sents.append(item["tweet"])

    df = pd.DataFrame(
        [(_, 1, "fr") for _ in sents], columns=["content", "toxic", "lang"]
    )
    df = df[["content", "toxic", "lang"]]

    return df


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    jigsaw_data = jigsaw()
    ru_comments = russian_language_toxic_comments()
    ru_comments_2 = toxic_russian_comments()
    clandestino_data = clandestino()
    germeval_data = germeval()
    rp_data = rp()
    mlma_data = mlma()

    merged = pd.concat(
        [
            jigsaw_data,
            ru_comments,
            ru_comments_2,
            clandestino_data,
            germeval_data,
            rp_data,
            mlma_data,
        ],
        ignore_index=True,
    )

    logging.info({l: len(merged[merged.lang == l]) for l in merged.lang.unique()})

    merged.to_csv("data/synth_paradetox_raw.csv", sep="\t", index=False)
