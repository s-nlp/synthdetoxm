import argparse

import datasets
import pandas as pd
import tqdm
import transformers
from transformers.pipelines.pt_utils import KeyDataset


def save_outputs(inputs, outputs, langs, save_path):
    df = pd.DataFrame(
        {
            'toxic_sentence': inputs,
            'neutral_sentence': outputs,
            'lang': langs,
        }
    )
    df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the generated outputs.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of return sequences for each input.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for generation.",
    )

    LANG_PROMPTS = {
        "es": "Write a non-toxic version of the following text in Spanish: ",
        "ru": "Write a non-toxic version of the following text in Russian: ",
        "de": "Write a non-toxic version of the following text in German: ",
        "fr": "Write a non-toxic version of the following text in French: ",
    }

    args = parser.parse_args()

    pipe = transformers.pipeline(
        "text2text-generation",
        model=args.model_path,
        device="cuda",
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
    )
    data = datasets.load_dataset("textdetox/multilingual_paradetox_test")
    
    inputs = []
    outputs = []
    langs = []
    for lang in data:
        if lang in LANG_PROMPTS.keys():
            inputs.extend(data[lang]['text'])
            # data_with_prompt = data[lang].map(lambda x: {"text": x["text"]})
            data_with_prompt = data[lang].map(lambda x: {'text': LANG_PROMPTS[lang] + x['text']})

            for out in tqdm.tqdm(
                pipe(
                    KeyDataset(data_with_prompt, "text"),
                    batch_size=args.batch_size,
                    truncation="only_first",
                )
            ):
                outputs.append(out[0]['generated_text'])
                langs.append(lang)

    save_outputs(inputs, outputs, langs, args.save_path)
