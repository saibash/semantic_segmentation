#!/bin/bash

PATH_TO_CHECKPOINT="./tmp/checkpoints"
PATH_TO_VIS_DIR="./tmp/vis"
PATH_TO_DATASET="./datasets/cityscapes/tfrecord"
PATH_TO_TRAIN_DIR="./tmp/train_dir"
PATH_TO_INITIAL_CHECKPOINT="./tmp/checkpoints"

python train.py \
    --logtostderr \
    --training_number_of_steps=90000 \
    --train_split="train_fine" \
    --train_crop_size="256,256" \
    --train_batch_size=1 \
    --dataset="cityscapes" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}