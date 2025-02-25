# !/bin/bash


BASE_MODEL_FOLDER="/disk/4tb/sushko/detox/"

# for batch in {0..9}; do
#     BATCH_MODEL_FOLDER="${BASE_MODEL_FOLDER}mt0_large_batch_${batch}"
#
#     CHECKPOINT_FOLDER=$(ls "${BATCH_MODEL_FOLDER}" | grep 'checkpoint-')
#
#     if [ -z "${CHECKPOINT_FOLDER}" ]; then
#         echo "No checkpoint folder found in ${BATCH_MODEL_FOLDER}"
#         continue
#     fi
#
#     MODEL_FOLDER="${BATCH_MODEL_FOLDER}/${CHECKPOINT_FOLDER}"
#     echo $MODEL_FOLDER
#     python3 src/run_inference_on_test.py \
#         --model_path "${MODEL_FOLDER}" \
#         --save_path "data/mt0_predictions/outputs_batch_${batch}.csv" \
#         --max_length 128 \
#         --num_return_sequences 1 \
#         --batch_size 64
# done
#
# BATCH_MODEL_FOLDER="${BASE_MODEL_FOLDER}mt0_xl_full_set/mt0_large_batch_0/"
# CHECKPOINT_FOLDER=$(ls "${BATCH_MODEL_FOLDER}" | grep 'checkpoint-')
# python3 src/run_inference_on_test.py \
#     --model_path "${BATCH_MODEL_FOLDER}/${CHECKPOINT_FOLDER}" \
#     --save_path "data/mt0_predictions/outputs_batch_full_set.csv" \
#     --max_length 128 \
#     --num_return_sequences 1 \
#     --batch_size 64
#
BATCH_MODEL_FOLDER="${BASE_MODEL_FOLDER}mt0_large_batch_full_with_golden/"
CHECKPOINT_FOLDER=$(ls "${BATCH_MODEL_FOLDER}" | grep 'checkpoint-')
python3 src/run_inference_on_test.py \
    --model_path "${BATCH_MODEL_FOLDER}/${CHECKPOINT_FOLDER}" \
    --save_path "data/mt0_predictions/outputs_batch_full_set_with_golden.csv" \
    --max_length 128 \
    --num_return_sequences 1 \
    --batch_size 64

BATCH_MODEL_FOLDER="${BASE_MODEL_FOLDER}mt0_xl_golden/"
CHECKPOINT_FOLDER=$(ls "${BATCH_MODEL_FOLDER}" | grep 'checkpoint-')
python3 src/run_inference_on_test.py \
    --model_path "${BATCH_MODEL_FOLDER}/${CHECKPOINT_FOLDER}" \
    --save_path "data/mt0_predictions/outputs_batch_golden.csv" \
    --max_length 128 \
    --num_return_sequences 1 \
    --batch_size 64

