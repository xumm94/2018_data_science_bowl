#!/bin/sh

export CUDA_VISIBLE_DEVICES=2

python ./train_nuclei.py \
    train \
    --model ./pretrained_model/mask_rcnn_coco.h5 \
    --ckpt ./expr/ \
    --dataset ./data/ \
    --epochs  40\
    --finetune all \
