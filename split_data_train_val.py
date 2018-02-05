"""
Split the dataset in ./data/stage1_train into training/val set with the raitio 9:1. 
The validation set will be saved in ./data/stage1_val

"""

import os, random, shutil

trainset_dir = "./data/stage1_train/"
valset_dir = "./data/stage1_val/"

if not os.path.exists(valset_dir):
    os.makedirs(valset_dir)

img_ids = next(os.walk(trainset_dir))[1]
val_img_ids = random.sample(img_ids, int(0.1*len(img_ids)))

for img_id in val_img_ids:
	shutil.move(trainset_dir+img_id, valset_dir+valset_dir)


