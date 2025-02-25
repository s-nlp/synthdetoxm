import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import dotenv
from googleapiclient import discovery
from nltk import sent_tokenize

dotenv.load_dotenv(".env")

API_KEY = f"{os.environ.get('PERSPECTIVE_API_KEY')}"

client = discovery.build(
    "commentanalyzer",
    "v1alpha1",
    developerKey=API_KEY,
    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
    static_discovery=False,
)

requested_attr = [
    "TOXICITY",
    "SEVERE_TOXICITY",
    "IDENTITY_ATTACK",
    "INSULT",
    "PROFANITY",
    # "THREAT",
]


def get_sta_scores(
    text: str,
    lang: Optional[str] = None,
    return_spans: Optional[bool] = True,
    toxicity_threshold: float = 0.5,
    return_full_response: bool = True,
) -> Dict[str, Optional[List[str]]]:
    """
    Sends a text to the Perspective API and returns sentences that are classified as toxic based on spans of attributes.

    Args:
        text (str): The text to be analyzed.
        lang (Optional[str]): The language of the text (optional). If None, language detection is applied.
        return_spans (bool): Whether to request span annotations in the API response.
        toxicity_threshold (float): Threshold for toxicity (profanity) for spans to consider.

    Returns:
        dict: A dictionary containing the original text, a list of toxic sentences, and the detected language.
              If an error occurs, the dictionary contains the error message.
    """
    analyze_request = {
        "comment": {"text": text},
        "requestedAttributes": {attr: {} for attr in requested_attr},
        "spanAnnotations": return_spans,
    }

    if lang:
        analyze_request["languages"] = [lang]

    try:
        response = client.comments().analyze(body=analyze_request).execute()
        toxic_spans = extract_toxic_spans(
            response, toxicity_threshold=toxicity_threshold
        )
        sentences = sent_tokenize(text)
        toxic_sentences = extract_toxic_sentences(text, sentences, toxic_spans)
        res = {
            "text": text,
            "toxic_sentences": toxic_sentences,
            "detected_lang": response.get("detectedLanguages", ["Unknown"]),
        }

        if return_full_response:
            res["response"] = response

        return res

    except Exception as e:
        return {"error": str(e)}


def extract_toxic_sentences_with_labels(
    text: str, sentences: List[str], response: Dict[str, Union[str, float]]
) -> List[Dict[str, Union[str, List[Tuple[int, int]]]]]:
    """
    Identifies sentences that overlap with toxic spans and returns them with their labels and spans.

    Args:
        text (str): The original text.
        sentences (list): A list of sentences extracted from the text.
        response (dict): The API response containing span scores for various attributes.

    Returns:
        list: A list of dictionaries, each containing a toxic sentence, its labels, and the spans of those labels.
    """
    toxic_sentences = []

    sentence_start_pos = 0
    for sentence in sentences:
        sentence_end_pos = sentence_start_pos + len(sentence)

        sentence_labels = {}
        for attr in requested_attr:
            if attr in response["attributeScores"]:
                for span in response["attributeScores"][attr].get("spanScores", []):
                    if (
                        span["begin"] < sentence_end_pos
                        and span["end"] > sentence_start_pos
                    ):
                        if attr not in sentence_labels:
                            sentence_labels[attr] = []
                        sentence_labels[attr].append((span["begin"], span["end"]))

        if sentence_labels:
            toxic_sentences.append(
                {
                    "sentence": sentence,
                    "labels": sentence_labels,
                }
            )

        sentence_start_pos = sentence_end_pos + 1

    return toxic_sentences


def extract_toxic_spans(
    response: Dict[str, Union[str, float]], toxicity_threshold: float = 0.5
) -> List[Tuple[int, int]]:
    """
    Extracts toxic spans from the Perspective API response where the score exceeds the given threshold.

    Args:
        response (dict): The API response containing span scores for various attributes.
        toxicity_threshold (float): The minimum score for a span to be considered toxic (default: 0.5).

    Returns:
        list: A list of tuples representing the start and end positions of the toxic spans.
    """
    toxic_attributes = ["TOXICITY", "SEVERE_TOXICITY", "THREAT", "PROFANITY"]
    toxic_spans = []

    for attr in toxic_attributes:
        if attr in response["attributeScores"]:
            for span in response["attributeScores"][attr].get("spanScores", []):
                if span["score"]["value"] >= toxicity_threshold:
                    toxic_spans.append((span["begin"], span["end"]))

    merged_spans = merge_spans(toxic_spans)
    return merged_spans


def merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Merges overlapping or adjacent spans into continuous spans.

    Args:
        spans (list): A list of tuples representing the start and end positions of spans.

    Returns:
        list: A list of merged spans, where overlapping or adjacent spans are combined.
    """
    if not spans:
        return []

    spans = sorted(spans, key=lambda x: x[0])
    merged_spans = [spans[0]]

    for current_start, current_end in spans[1:]:
        last_start, last_end = merged_spans[-1]

        if current_start <= last_end:
            merged_spans[-1] = (last_start, max(last_end, current_end))
        else:
            merged_spans.append((current_start, current_end))

    return merged_spans


def extract_toxic_sentences(
    text: str, sentences: List[str], toxic_spans: List[Tuple[int, int]]
) -> List[str]:
    """
    Identifies sentences that overlap with toxic spans.

    Args:
        text (str): The original text.
        sentences (list): A list of sentences extracted from the text.
        toxic_spans (list): A list of tuples representing the start and end positions of toxic spans.

    Returns:
        list: A list of sentences that overlap with any of the toxic spans.
    """
    toxic_sentences = []

    sentence_start_pos = 0
    for sentence in sentences:
        sentence_end_pos = sentence_start_pos + len(sentence)

        if any(
            span_start < sentence_end_pos and span_end > sentence_start_pos
            for span_start, span_end in toxic_spans
        ):
            toxic_sentences.append(sentence)

        sentence_start_pos = sentence_end_pos + 1

    return toxic_sentences
