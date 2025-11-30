#!/bin/bash

# Create dataset with 300 augmentations
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-300 --subsample-size 1000 --num-aug 300

# Train (adjust data_path to match)
OMP_NUM_THREADS=8 python pretrain.py data_path=data/sudoku-extreme-1k-aug-300 epochs=20000 eval_interval=2000 global_batch_size=384 lr=7e-5 puzzle_emb_lr=7e-5 weight_decay=1.0 puzzle_emb_weight_decay=1.0