import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import src.model as modellib


from src.CellsDataset import CellsConfig, CellsDataset
import src.visualize as visualize

# Root directory of the project
ROOT_DIR = os.getcwd()
print(ROOT_DIR)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

IMAGES_DIR = r'/home/futura/PycharmProjects/Kaggle/DSB_data/stage1_train'

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)

config = CellsConfig()
# config.display()


if __name__ == "__main__":
    dataset_train = CellsDataset()
    # dataset_train.load_cells(IMAGES_DIR, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_train.load_cells(IMAGES_DIR)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellsDataset()
    # dataset_val.load_cells(IMAGES_DIR, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
    dataset_val.load_cells(IMAGES_DIR)
    dataset_val.prepare()

    # Load and display random samples
    image_ids = np.random.choice(dataset_train.image_ids, 1)
    print(image_ids)
    for image_id in image_ids:
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)



    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # from keras.utils import plot_model
    # plot_model(model, to_file='model_training_mode.png', show_shapes=True)

    # Which weights to start with?
    init_with = "imagenet"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    print('Start train')

    model.train(dataset_train, dataset_val,
                # learning_rate=config.LEARNING_RATE,
                learning_rate=0.001,
                epochs=2,
                layers='heads')
