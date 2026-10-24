#!/bin/bash
# FedDASW-v5 (CMA + Momentum Queue) on Flickr30K

cd "$(dirname "$0")"

DATASET_DIR="${DATASET_DIR:-your_path/Flickr30K}"
GPU_ID="${GPU_ID:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

NUM_CLIENTS=10
FL_ROUNDS=10
LOCAL_EPOCHS=5
DIRICHLET_ALPHA=0.5
QUEUE_SIZE=4096
QUEUE_START_SIZE=4096
QUEUE_LOSS_WEIGHT=0.001
CLIENT_NOISE_RATIOS="0.0,0.2,0.2,0.4,0.4,0.4,0.6,0.6,0.6,0.6"

"${PYTHON_BIN}" src/main_fl.py \
    --name "fl_ours_flickr_queuew001" \
    --fed_method feddasw_v2 \
    --train_mode prompt \
    --num_clients ${NUM_CLIENTS} \
    --fl_rounds ${FL_ROUNDS} \
    --local_epochs ${LOCAL_EPOCHS} \
    --dirichlet_alpha ${DIRICHLET_ALPHA} \
    --client_noise_ratios "${CLIENT_NOISE_RATIOS}" \
    --fl_eval_freq 1 \
    --train_data "${DATASET_DIR}/train.csv" \
    --retrieval_test_data "${DATASET_DIR}/test_retrieval.csv" \
    --image_key image_path \
    --batch_size 64 \
    --lr 5e-5 \
    --device gpu \
    --device_id "${GPU_ID}" \
    --num_workers "${NUM_WORKERS}" \
    --queue_size ${QUEUE_SIZE} \
    --queue_start_size ${QUEUE_START_SIZE} \
    --queue_loss_weight ${QUEUE_LOSS_WEIGHT} \
    --logs ./logs \
    "$@"
