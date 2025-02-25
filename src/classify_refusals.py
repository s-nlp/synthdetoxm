import os
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Initialize the classification pipeline
classifier = pipeline('text-classification', model='chameleon-lizard/xlmr-base-refusal-classifier', device='cuda')

# Define the base directory for detox data
base_dir = './data/predictions_for_paper/'

# Define the base directory for toxic data
toxic_base_dir = './data/'

# Supported languages
languages = ['de', 'es', 'fr', 'ru']

# Mapping labels to integers
label_mapping = {
    'REFUSAL': 1,
    'NO_REFUSAL': 0
}

# Iterate over each language
for lang in languages:
    lang_dir = os.path.join(base_dir, lang)
    toxic_file_path = os.path.join(toxic_base_dir, f"toxic_clean_{lang}.txt")

    if not os.path.isdir(lang_dir):
        print(f"Directory {lang_dir} does not exist. Skipping {lang}.")
        continue

    if not os.path.isfile(toxic_file_path):
        print(f"Toxic file {toxic_file_path} does not exist. Skipping {lang}.")
        continue

    # Read toxic sentences
    with open(toxic_file_path, 'r') as f:
        toxic_sentences = [line.strip() for line in f]

    # Iterate over each file in the language directory (detox data)
    for filename in os.listdir(lang_dir):
        file_path = os.path.join(lang_dir, filename)

        # Skip if it's not a file or if it's 'concat.csv'
        if not os.path.isfile(file_path) or filename == 'concat.csv':
            continue

        # Read detox sentences
        try:
            if filename.endswith('.tsv'):
                continue
            elif filename.endswith('.csv'):
                continue
            else:
                with open(file_path, 'r') as f:
                    sentences = [line.strip() for line in f]
        except Exception as e:
            print(f"Error reading {file_path}: {e}. Skipping.")
            continue

        if not sentences:
            print(f"No sentences found in {file_path}. Skipping.")
            continue

        # Classify detox sentences
        batch_size = 32
        detox_results = []
        for i in tqdm(range(0, len(sentences), batch_size), desc=f"Processing detox: {filename}"):
            batch = sentences[i:i + batch_size]
            batch = [_[:500] for _ in batch]
            classifications = classifier(batch)
            for sentence, classification in zip(batch, classifications):
                label = classification['label']
                score = classification['score']
                mapped_label = label_mapping.get(label, 0)
                detox_results.append({'detox_sentence': sentence, 'detox_refusal': mapped_label})

        df_detox_classified = pd.DataFrame(detox_results)

        # Classify toxic sentences
        toxic_results = []
        for i in tqdm(range(0, len(toxic_sentences), batch_size), desc=f"Processing toxic: {lang}"):
            batch = toxic_sentences[i:i + batch_size]
            batch = [_[:500] for _ in batch]
            classifications = classifier(batch)
            for sentence, classification in zip(batch, classifications):
                label = classification['label']
                score = classification['score']
                mapped_label = label_mapping.get(label, 0)
                toxic_results.append({'toxic_sentence': sentence, 'toxic_refusal': mapped_label})

        df_toxic_classified = pd.DataFrame(toxic_results)

        # Ensure both DataFrames have the same length for merging
        min_len = min(len(df_detox_classified), len(df_toxic_classified))
        df_detox_classified = df_detox_classified.iloc[:min_len].reset_index(drop=True)
        df_toxic_classified = df_toxic_classified.iloc[:min_len].reset_index(drop=True)

        # Concatenate DataFrames
        df_combined = pd.concat([df_detox_classified, df_toxic_classified], axis=1)

        # Create the overall_refusal column
        df_combined['overall_refusal'] = (df_combined['detox_refusal'] == 1) & (df_combined['toxic_refusal'] == 0)

        # Define the output file path with .tsv extension
        output_filename = os.path.splitext(filename)[0] + '_classified.tsv'
        output_path = os.path.join(lang_dir, output_filename)

        # Save the combined DataFrame to TSV
        df_combined.to_csv(output_path, sep='\t', index=False)

        print(f"Saved classifications to {output_path}")
