import argparse
import logging
import pathlib

import datasets
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser.add_argument(
        "--final_folder_path",
        type=str,
        required=False,
        help="Path to final folder.",
        default="data/predictions_for_paper/",
    )

    parser.add_argument(
        "--calculate_statistics_by_model",
        required=False,
        action="store_true",
        help="Whether do display language acceptance statistics by model.",
    )

    args = parser.parse_args()

    final_folder_path = pathlib.Path(args.final_folder_path)

    all_dfs = []
    for lang_folder in final_folder_path.iterdir():
        lang = (lang_folder.name)

        if lang in ["ru", "de", "es", "fr"]:
            print(f"Processing language: {lang}")

            for file in lang_folder.iterdir():
                if (
                    file.name == "concat_new_final.csv"
                ):

                    df = pd.read_csv(file)

                    toxic_list = []
                    scores = []
                    toxic_sta_list_api = []
                    toxic_sta_list_pipe = []
                    detoxed_list = []
                    sta_list_api = []
                    sta_list_pipe = []
                    sim_list = []
                    which_model = []
                    all_refused_list = []
                    is_detoxifiable_list_api = []
                    is_detoxifiable_list_pipe = []
                    reduction_list_api = []
                    reduction_list_pipe = []

                    detoxed_cols = [
                        col for col in df.columns if col.startswith("detoxed_")
                    ]
                    sta_cols = [
                        col
                        for col in df.columns
                        if col.startswith("sta_") and col != "sta_toxic_api" and col != 'sta_toxic_pipe'
                    ]
                    sim_cols = [col for col in df.columns if col.startswith("sim_")]

                    group_ids = list(set(col.split("_", 1)[1] for col in detoxed_cols if 'llama' not in col.lower()))

                    for idx, row in df.iterrows():
                        toxic_list.append(row["toxic_text"])
                        toxic_sta_list_api.append(1 - row["sta_toxic_api"])
                        toxic_sta_list_pipe.append(1 - row["sta_toxic_pipe"])

                        best_detoxed = None
                        best_sta_api = None
                        best_sta_pipe = None
                        best_sim = None
                        best_score = -float("inf")

                        best_col = group_ids[0]

                        all_refused = sum([row[f'is_refusal_{_}'] for _ in group_ids]) > len(group_ids) - 1
                        for group_id in group_ids:
                            if not all_refused:
                                if row[f'is_refusal_{group_id}']:
                                    continue

                            detoxed_col = f"detoxed_{group_id}"
                            sta_col_api = f"sta_detoxed_api_{group_id}"
                            sta_col_pipe = f"sta_detoxed_pipe_{group_id}"
                            sim_col = f"sim_detoxed_{group_id}"

                            if row['toxic_text'] is None or row[detoxed_col] is None:
                                continue

                            if pd.notnull(row[sta_col_api]) and pd.notnull(row[sta_col_pipe]) and pd.notnull(row[sim_col]):
                                score = abs(row['sta_toxic_api'] - row[sta_col_api]) * abs(row['sta_toxic_pipe'] - row[sta_col_pipe]) * row[sim_col]
                                if score > best_score:
                                    best_score = score
                                    best_detoxed = row[detoxed_col]
                                    best_sta_api = row[sta_col_api]
                                    best_sta_pipe = row[sta_col_pipe]
                                    best_sim = row[sim_col]
                                    best_col = group_id
                        
                        scores.append(score)
                        detoxed_list.append(best_detoxed)
                        sta_list_api.append(best_sta_api)
                        sta_list_pipe.append(best_sta_pipe)
                        sim_list.append(best_sim)
                        which_model.append(best_col)
                        all_refused_list.append(all_refused)
                        reduction_list_api.append(row['sta_toxic_api'] - best_sta_api)
                        reduction_list_pipe.append(row['sta_toxic_pipe'] - best_sta_pipe)
                        is_detoxifiable_list_api.append(row['detoxifiable_api'])
                        is_detoxifiable_list_pipe.append(row['detoxifiable_pipe'])

                    lang_df = pd.DataFrame(
                        {
                            "Toxic": toxic_list,
                            "Toxic STA api": toxic_sta_list_api,
                            "Toxic STA pipe": toxic_sta_list_pipe,
                            "Detoxed": detoxed_list,
                            "Score": scores,
                            "STA api": sta_list_api,
                            "STA pipe": sta_list_pipe,
                            "SIM": sim_list,
                            "Lang": lang,
                            "all_refused": all_refused_list,
                            "detoxifiable_api": is_detoxifiable_list_api,
                            "detoxifiable_pipe": is_detoxifiable_list_pipe,
                            'reduction_list_api': reduction_list_api,
                            'reduction_list_pipe': reduction_list_pipe,
                            "Which model": which_model,
                        }
                    )

                    lang_df = lang_df.sort_values(by=['detoxifiable_pipe', 'reduction_list_pipe'], ascending=False)
                    all_dfs.append(lang_df)

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv("data/predictions_for_paper/final_data_full.tsv", sep='\t', index=False)

    hf_dataset = datasets.Dataset.from_pandas(final_df)

    hf_dataset.save_to_disk(
        "data/predictions_for_paper/synthetic_paradetox"
    )

    if args.calculate_statistics_by_model:
        final_df["model"] = final_df["Which model"].apply(lambda x: x[:-7 - len('_classified')])
        final_df_sorted = final_df
        final_df_filtered = final_df_sorted.groupby("Lang").head(4000)

        logging.info(final_df_filtered.groupby("model")["Lang"].count().to_markdown())
        logging.info(final_df_filtered.groupby(["model", "Lang"]).size().to_markdown())
