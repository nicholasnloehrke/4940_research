#!/bin/bash

RUN_ID=$(date +"%Y%m%d_%H%M%S")

LOG_PATH="runs/run_$RUN_ID"

# python rnn_wip.py data/embeddings_rall_pnone_no_tokenizer.pkl --hidden_size 128 --layers 1 --batch_size 1 --epochs 30 --learning_rate 0.0005 --folds 5 --bidirectional --log_path "$LOG_PATH"
python rnn.py data/augmented_embeddings.pkl --hidden_size 128 --layers 1 --batch_size 1 --epochs 20 --learning_rate 0.0005 --folds 5
