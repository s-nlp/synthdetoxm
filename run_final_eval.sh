# !/bin/bash

#
# for batch in {0..9}; do
#     BATCH_MODEL_FILE="data/mt0_predictions/outputs_batch_${batch}.csv"
#
#     echo $BATCH_MODEL_FILE
#     python3 src/evaluate_on_multiparadetox.py \
#         --test_data_path "data/predictions_for_paper/multiparadetox_test.csv" \
#         --predictions_path "${BATCH_MODEL_FILE}" \
#         --output_base "results/batch_${batch}_" 
# done
#
# python3 src/evaluate_on_multiparadetox.py \
#     --test_data_path "data/predictions_for_paper/multiparadetox_test.csv" \
#     --predictions_path "data/mt0_predictions/outputs_batch_full_set.csv" \
#     --output_base "results/full_set_evaluation_results"
#
python3 src/evaluate_on_multiparadetox.py \
    --test_data_path "data/predictions_for_paper/multiparadetox_test.csv" \
    --predictions_path "data/mt0_predictions/outputs_batch_full_set_with_golden.csv" \
    --output_base "results/full_set_gold_evaluation_results"

python3 src/evaluate_on_multiparadetox.py \
    --test_data_path "data/predictions_for_paper/multiparadetox_test.csv" \
    --predictions_path "data/mt0_predictions/outputs_batch_golden.csv" \
    --output_base "results/gold_evaluation_results"

