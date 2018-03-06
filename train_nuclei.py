# coding: utf-8

"""
Mask R-CNN - Train on Nuclei Dataset (Updated from train_shape.ipynb)

This notebook shows how to train Mask R-CNN on your own dataset. 
To keep things simple we use a synthetic dataset of shapes (squares, 
triangles, and circles) which enables fast training. You'd still 
need a GPU, though, because the network backbone is a Resnet101, 
which would be too slow to train on a CPU. On a GPU, you can start 
to get okay-ish results in a few minutes, and good results in less than an hour.
"""



import os
import sys
import random
import math
import re
import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log
import logging
import argparse

"""
Configurations

Override form Config
"""

class NucleiConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 300

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

    LEARNING_RATE = 0.001

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    def display(self, logger):
        """Display Configuration values."""
        print("\nConfigurations:")
        logger.info('\nConfigurations:')
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
                logger.info("{:30} {}".format(a, getattr(self, a)))
        print("\n")

class NucleiDataset(utils.Dataset):

    """Load the images and masks from dataset."""

    def load_image_info(self, data_path, img_set = None):
        """Get the picture names(ids) of the dataset."""
        
        # Add classes
        self.add_class("nucleis", 1, "regular")
        # TO DO : Three different image types into three classes
        
        # Add images
        # Get the images ids of training/testing set
        if img_set is None:
            train_ids = next(os.walk(data_path))[1]
        else:
            with open(img_set) as f:
                read_data = f.readlines()
            train_ids = [read_data[i][:-1] for i in range(0,len(read_data))] # Delete New line '\n'
        # Get the info of the images
        for i, id_ in enumerate(train_ids):
            file_path = os.path.join(data_path, id_)
            img_path = os.path.join(file_path, "images")
            masks_path = os.path.join(file_path, "masks")
            img_name = id_ + ".png"
            img = cv2.imread(os.path.join(img_path, img_name))
            width, height, _ = img.shape
            self.add_image("nucleis", image_id=id_, path=file_path,
                           img_path=img_path, masks_path=masks_path,
                           width=width, height=height,
                           nucleis="nucleis")                

    def load_image(self, image_id):
        """Load image from file of the given image ID."""
        info = self.image_info[image_id]
        img_path = info["img_path"]
        img_name = info["id"] + ".png"
        image = cv2.imread(os.path.join(img_path, img_name))
        return image

    def image_reference(self, image_id):
        """Return the path of the given image ID."""
        info = self.image_info[image_id]
        if info["source"] == "nucleis":
            return info["path"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load the instance masks of the given image ID."""
        info = self.image_info[image_id]
        mask_files = next(os.walk(info["masks_path"]))[2]
        masks = np. zeros([info['width'], info['height'], len(mask_files)], dtype=np.uint8)
        for i, id_ in enumerate(mask_files):
            single_mask = cv2.imread(os.path.join(info["masks_path"], id_), 0)
            masks[:, :, i:i+1] = single_mask[:, :, np.newaxis]
        class_ids = np.ones(len(mask_files))
        return masks, class_ids.astype(np.int32)

    # def test(self):
    #     return "1"

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def parser_argument():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Nuclei Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' or predict")
    parser.add_argument('--datapath',
                        metavar="/path/to/data/",
                        default="./data",
                        help='Directory of the Nuclei dataset')
    parser.add_argument('--init_with',
                        metavar="/init/type",
                        default="coco",
                        help="Initialize with the (\"coco\"/\"imagenet\"/\"last\") net")
    parser.add_argument('--model',
                        metavar="/path/to/weights.h5",
                        default="./models/mask_rcnn_coco.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--ckpt',
                        metavar="/path/to/save/checkpoint",
                        default="/data/lf/Nuclei/logs",
                        help="Directory of the checkpoint")
    parser.add_argument('--epochs',
                        metavar="/num/of/epochs",
                        default="50",
                        help="The number of the training epochs")
    parser.add_argument('--finetune',
                        metavar="/finetune/type",
                        default="heads",
                        help="The type of the finetune method(\"heads\" or \"all\")")
    parser.add_argument('--lr_start',
                        metavar="/value/of/start/lr",
                        default="0.001",
                        type=float,
                        help="The Value of learning rate to start")
    parser.add_argument('--train_dataset',
                        metavar="train/imgs/names",
                        default="10-fold-train-1.txt",
                        help="The training set split of the data")
    parser.add_argument('--val_dataset',
                        metavar="val/imgs/names",
                        default="10-fold-val-1.txt",
                        help="The validation set split of the data")

    return parser.parse_args()

if __name__ == '__main__':
    
    args = parser_argument()
    logname = "config-" + time.strftime('%Y%m%d%H%M', time.localtime(time.time())) +".log"
    logging.basicConfig(filename=os.path.join(args.ckpt, logname), level=logging.INFO)
    logger = logging.getLogger('root')
    logger.info('\nBasic Setting:')
    logger.info('\nCommand: {} \n Initialize: {} \n Model: {} \n Datapath: {} \n Ckpt: {} \n Epochs \
        : {} \n Finetune: {} \n Train_dataset: {} \n Val_dataset: {} \n' \
        .format(args.command, args.init_with, args.model, args.datapath, args.ckpt,\
            args.epochs, args.finetune, args.train_dataset, args.val_dataset))

    # Train or evaluate or predict
    if args.command == "train":

        config = NucleiConfig()
        config.LEARNING_RATE = args.lr_start
        config.display(logger)
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.ckpt)
        
        # Select weights file to load
        print("Loading weights From ", args.model)

        if args.init_with == "imagenet":
            model.load_weights(model.get_imagenet_weights(), by_name=True)
        elif args.init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
            model.load_weights(args.model, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
        elif args.init_with == "last":
        # Load the last model you trained and continue training
            model.load_weights(args.model, by_name=True)

        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        DATASET_DIR = os.path.join(args.datapath, "stage1_train_fixed")
        TRAIN_IMG_SET = os.path.join(args.datapath, "stage1_train_fixed_10fold", args.train_dataset)
        VAL_IMG_SET = os.path.join(args.datapath, "stage1_train_fixed_10fold", args.val_dataset)

        dataset_train = NucleiDataset()
        dataset_train.load_image_info(DATASET_DIR, TRAIN_IMG_SET)
        dataset_train.prepare()

        dataset_val = NucleiDataset()
        dataset_val.load_image_info(DATASET_DIR, VAL_IMG_SET)
        dataset_val.prepare()

        print("Loading {} training images, {} validation images"
              .format(len(dataset_train.image_ids), len(dataset_val.image_ids)))


        if args.finetune == "heads":
            model.train(dataset_train, dataset_val, 
                        learning_rate=config.LEARNING_RATE, 
                        epochs=int(args.epochs), 
                        layers='heads',
                        logger=logger)
        elif args.finetune == "all":
            model.train(dataset_train, dataset_val,
                        learning_rate=config.LEARNING_RATE,
                        epochs=int(args.epochs),
                        layers='all',
                        logger=logger)
        else: 
            raise NameError("Only two finetune type is vaild(\"heads\" or \"all\")")


    elif args.command == "evaluate": 
        # TODO AP in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        class InferenceConfig(NucleiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MAX_INSTANCES = 300
        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.ckpt)

        print("Loading weights From ", args.model)
        model.load_weights(args.model, by_name=True)

        VALSET_DIR = os.path.join(args.dataset, "stage1_val")
        dataset_val = NucleiDataset()
        dataset_val.load_image_info(VALSET_DIR)
        dataset_val.prepare()
        print("Evaluate {} images".format(len(dataset_val.image_ids)))

        APs = []

        for image_id in tqdm(dataset_val.image_ids):
            # Load image and ground truth data
            image, image_meta, gt_class_id, gt_bbox, gt_mask =modellib.load_image_gt(
                dataset_val, InferenceConfig, image_id, use_mini_mask=False)
            molded_images = np.expand_dims(modellib.mold_image(image, config), 0)

            # Run object detection
            results = model.detect([image], verbose=0)
            r = results[0]

            # Compute AP
            AP, precisions, recalls, overlaps =utils.compute_ap(gt_bbox, 
                gt_class_id,r["rois"], r["class_ids"], r["scores"], iou_threshold=0.5)
            APs.append(AP)

        print("mAP: ", np.mean(APs))

    elif args.command == "predict":

        class InferenceConfig(NucleiConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_NMS_THRESHOLD = 0.3
            DETECTION_MAX_INSTANCES = 300

        config = InferenceConfig()
        config.display()

        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.ckpt)

        print("Loading weights From ", args.model)
        model.load_weights(args.model, by_name=True)

        TESTSET_DIR = os.path.join(args.dataset, "stage1_test")
        dataset_test = NucleiDataset()
        dataset_test.load_image_info(TESTSET_DIR)
        dataset_test.prepare()

        print("Predict {} images".format(dataset_test.num_images))

        test_ids = []
        test_rles = []

        for image_id in tqdm(dataset_test.image_ids):
            image = dataset_test.load_image(image_id)
            id_ = dataset_test.image_info[image_id]["id"]
            results = model.detect([image], verbose=0)
            r = results[0]
            mask_exist = np.zeros(r['masks'].shape[:-1], dtype=np.uint8)
            for i in range(r['masks'].shape[-1]):
                _mask = r['masks'][:,:,i]
                overlap_index = np.where(np.multiply(mask_exist, _mask) == 1)
                _mask[overlap_index] = 0
                mask_exist += _mask
                if np.any(_mask):
                    test_ids.append(id_)
                    test_rles.append(rle_encoding(_mask))
                else :
                    continue
                # if np.count_nonzero(_mask) > 40 :
                #     test_ids.append(id_)
                #     test_rles.append(rle_encoding(_mask))
                # else :
                #     continue

        sub = pd.DataFrame()
        sub['ImageId'] = test_ids
        sub['EncodedPixels'] = pd.Series(test_rles).apply(lambda x: ' '.join(str(y) for y in x))
        csvpath = "{}.csv".format(args.model)
        print("Writing the Result in {}".format(csvpath))
        sub.to_csv(csvpath, index=False)

    else:
        print("'{}' is not recognized. Use 'train' 'evaluate' 'predict'".format(args.command))


