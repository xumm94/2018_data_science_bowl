import os
import argparse
import random
import numpy as np
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--fold_num', default='10', type=int)
ap.add_argument('--data_dir', default='../data/stage1_train_fixed')
args = ap.parse_args()


random.seed()
image_dir = args.data_dir
image_list = os.listdir(image_dir)
random.shuffle(image_list)
img_nums = len(image_list)
kfold_img_nums = int(img_nums/args.fold_num)
print('Total images {} | Each fold images {} '.format(img_nums, kfold_img_nums))
folds = [image_list[i:i+kfold_img_nums] for i in range(0,len(image_list),kfold_img_nums)]
print('Fold shape: {}'.format(folds[0]))
for i in range(args.fold_num):
    print('Processing Fold {}'.format(i))
    train_fn = '{}-fold-train-{}.txt'.format(args.fold_num, i+1)
    val_fn = '{}-fold-val-{}.txt'.format(args.fold_num, i+1)
    with open(train_fn, 'w') as ftrain, open(val_fn, 'w') as fval:

        count = 0
        for j in range(args.fold_num):
            if j == i:
                for line in folds[j]:
                    fval.write('{}'.format(line))
                    fval.write('\n')
                    count += 1
            else:
                for line in folds[j]:
                    ftrain.write('{}'.format(line))
                    ftrain.write('\n')
                    count += 1
            print('Processing Fold {} | {} images processed'.format(i, count))

