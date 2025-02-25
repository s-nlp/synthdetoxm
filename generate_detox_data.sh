#!/bin/bash
models=(
    "unsloth/Llama-3.3-70B-Instruct"
    "CohereForAI/c4ai-command-r-08-2024"
    "mistralai/Mistral-Small-Instruct-2409"
    "Qwen/Qwen2.5-32B-Instruct"
    "microsoft/Phi-3-mini-4k-instruct"
    "microsoft/Phi-3-medium-4k-instruct"
    "mistralai/Mistral-Nemo-Instruct-2407"
    "google/gemma-2-9b-it"
    "google/gemma-2-27b-it"
)

languages=("de" "fr" "ru" "es")

for model in "${models[@]}"; do
    for lang in "${languages[@]}"; do
        echo "Processing language: $lang"
        echo "Using model: $model"

        python3 src/generate.py \
            --model "$model" \
            --lang "$lang" \
            --data_path "data/toxic_clean_${lang}.txt" \
            --output_path "generated/${lang}" \
            --batch_size 1 \
            --temp 0.9
    done
done