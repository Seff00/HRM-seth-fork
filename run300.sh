#!/bin/bash

DATA_DIR="data/sudoku-extreme-1k-aug-300"

if [ ! -d "$DATA_DIR" ]; then
    echo "Dataset not found. Generating..."
    python dataset/build_sudoku_dataset.py --output-dir "$DATA_DIR" --subsample-size 1000 --num-aug 300
else
    echo "Dataset found at $DATA_DIR. Skipping generation."
fi

# Train (adjust data_path to match)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-300 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0