#!/bin/sh

export CUDA_VISIBLE_DEVICES=4,5

for ((i=1; i<=10; i++))
do
    train_dataset=10-fold-train-$i.txt
    val_dataset=10-fold-val-$i.txt

    python ./train_nuclei.py \
        train \
        --model ./models/mask_rcnn_coco.h5 \
        --ckpt /data/lf/Nuclei/logs-all \
        --datapath ./data/ \
        --epochs  50\
        --finetune all \
        --lr_start 0.005 \
        --train_dataset $train_dataset \
        --val_dataset $val_dataset

done
