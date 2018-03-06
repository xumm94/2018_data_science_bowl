#!/bin/sh

export CUDA_VISIBLE_DEVICES=6,7

for ((i=1; i<=10; i++))
do
    train_dataset=10-fold-train-$i.txt
    val_dataset=10-fold-val-$i.txt

    python ./train_nuclei.py \
        train \
        --model ./models/mask_rcnn_coco.h5 \
        --ckpt /data/lf/Nuclei/logs-head \
        --datapath ./data/ \
        --epochs  50\
        --finetune heads \
        --lr_start 0.01 \
        --train_dataset $train_dataset \
        --val_dataset $val_dataset

done