import argparse
import os
import re
from copy import deepcopy

from dotenv import load_dotenv
from tqdm.auto import trange
from vllm import LLM, SamplingParams

from utils.prompting import compose_prompt


def detoxify_text(args):
    llm = LLM(model=args.model, max_model_len=4092, trust_remote_code=True)

    with open(args.data_path, "r") as f:
        data_rus = f.read().split("\n")

    conversation = compose_prompt(model_name=args.model, language=args.lang)

    if args.model == "Qwen/Qwen2.5-32B-Instruct":
        sampling_params = SamplingParams(
            temperature=0.7, top_p=0.95, min_p=0.05, top_k=40, max_tokens=512
        )
    elif args.model in (
        "mistralai/Mistral-Small-Instruct-2409",
        "mistralai/Mistral-Nemo-Instruct-2407",
    ):
        sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
    else:
        sampling_params = SamplingParams(
            temperature=args.temp or 0.9, top_p=0.9, max_tokens=512
        )

    output_file = os.path.join(
        args.output_path, f"{args.model.split('/')[-1]}_{args.lang}"
    )

    preds = []
    for i in trange(0, len(data_rus), args.batch_size):
        batch = data_rus[i : i + args.batch_size]

        conv_batch = []
        for toxic_text in batch:
            conv_ = deepcopy(conversation)
            conv_.append({"role": "user", "content": toxic_text})
            conv_batch.append(conv_)

        outputs = llm.chat(
            messages=conv_batch, sampling_params=sampling_params, use_tqdm=False
        )

        for output in outputs:
            preds.append(re.sub("\n", "", output.outputs[0].text))

    with open(output_file, "w") as f:
        f.write("\n".join(preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detoxify toxic text using a selected model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-Small-Instruct-2409",
        help="The model to use for detoxification.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for processing."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/toxic_clean_es.txt",
        help="Path to the input file containing toxic texts.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data",
        help="Directory to save the detoxified text.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.7,
        required=False,
        help="Temperature for generation.",
    )

    parser.add_argument(
        "--lang", type=str, required=False, default="ru", help="Language."
    )
    args = parser.parse_args()

    load_dotenv()

    detoxify_text(args)
