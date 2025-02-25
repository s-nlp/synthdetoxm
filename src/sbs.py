import argparse
import os

import dotenv
import httpx
import openai
import pandas as pd
import tqdm

from httpx_socks import SyncProxyTransport

import utils.prompting


dotenv.load_dotenv(".env")

def send_question(
    prompt: str,
    model: str,
    api_link: str,
    token: str,
    temperature: float,
    max_tokens: int,
):
    transport = SyncProxyTransport.from_url('socks5://user:passwd@ip:port')
    http_client = httpx.Client(transport=transport)

    client = openai.OpenAI(
        http_client=http_client,
        api_key=token,
    )

    messages = []
    messages.append({"role": "user", "content": prompt})

    response_big = None
    idx = 0
    while response_big is None or idx == 10:
        try:
            response_big = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
            )
            idx += 1
        except openai.APIConnectionError:
            idx += 1

    if idx == 10:
        return "This sentence was not judged."

    response = response_big.choices[0].message.content

    return response


def _main():
    parser = argparse.ArgumentParser(description='Process two file paths.')

    # Add arguments
    parser.add_argument('--path_to_first_file', type=str, required=True,
                        help='The path to the first file.')
    parser.add_argument('--path_to_second_file', type=str, required=True,
                        help='The path to the second file.')
    parser.add_argument('--result_path', type=str, required=True,
                        help='The path to results.')

    # Parse the arguments
    args = parser.parse_args()

    first_data = pd.read_csv(args.path_to_first_file)
    second_data = pd.read_csv(args.path_to_second_file)

    judge_model = f"{os.environ.get('JUDGE_MODEL')}"
    judge_api_link = f"{os.environ.get('JUDGE_API_LINK')}"
    token = f"{os.environ.get('TOKEN')}"

    text_to_score = {
        'A': 1,
        'B': -1,
        'Tie': 0,
    }
    
    sources = []
    q1 = []
    q2 = []
    language = []
    res_q1_vs_q2 = []
    res_q2_vs_q1 = []
    for first, second in tqdm.tqdm(zip(first_data.iterrows(), second_data.iterrows())):
        first = first[1]
        second = second[1]

        prompt_1, prompt_2 = utils.prompting.compose_sbs_prompt(
            toxic_sentence=first.source_text,
            query_1=first.detoxified_text,
            query_2=second.detoxified_text,
            language=first.language,
        )

        q1_vs_q2 = send_question(
            prompt=prompt_1,
            model=judge_model,
            api_link=judge_api_link,
            token=token,
            temperature=0.0,
            max_tokens=512,
        )

        q2_vs_q1 = send_question(
            prompt=prompt_2,
            model=judge_model,
            api_link=judge_api_link,
            token=token,
            temperature=0.0,
            max_tokens=512,
        )

        q1_vs_q2 = text_to_score[q1_vs_q2]
        q2_vs_q1 = text_to_score[q2_vs_q1]

        sources.append(first.source_text)
        q1.append(first.detoxified_text)
        q2.append(second.detoxified_text)
        language.append(first.language)
        res_q1_vs_q2.append(q1_vs_q2)
        res_q2_vs_q1.append(q2_vs_q1)

    res = pd.DataFrame(
        {
            'sources': sources,
            'q1': q1,
            'q2': q2,
            'language': language,
            'q1_vs_q2': res_q1_vs_q2,
            'q2_vs_q1': res_q2_vs_q1,
        }
    )

    res.to_csv(args.result_path, sep='\t', index=False)

    print(f'SBS {args.path_to_first_file} vs {args.path_to_second_file}:')
    print(res.describe())
    res.q2_vs_q1 = res.q2_vs_q1 * -1


if __name__ == '__main__':
    _main()
