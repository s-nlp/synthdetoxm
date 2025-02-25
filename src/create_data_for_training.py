import pandas as pd


df_full = pd.read_csv('data/predictions_for_paper/final_data_full.tsv', sep='\t')
test = pd.read_csv('data/predictions_for_paper/test_without_answers.tsv', sep='\t')

to_remove = df_full['Toxic'].isin(test['toxic_sentence'])
df_full_filtered = df_full[~to_remove]

df_grouped = df_full_filtered.groupby('Lang').apply(lambda x: x.head(4000)).reset_index(drop=True)

df_grouped = df_grouped[df_grouped['Lang'] != 'fr']

df_full_filtered.to_csv('data/predictions_for_paper/final_data_full_noleak.tsv', sep='\t', index=False)
df_grouped.to_csv('data/predictions_for_paper/train_no_leak.tsv', sep='\t', index=False)
