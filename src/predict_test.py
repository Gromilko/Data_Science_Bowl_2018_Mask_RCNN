import src.model as modellib
from src.CellsDataset import CellsConfig, CellsDataset
import src.visualize as visualize
import random
from src.model import log
import matplotlib
import matplotlib.pyplot as plt

import skimage.io

import os
import numpy as np
import src.utils as utils


ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

IMAGES_DIR = r'/home/futura/PycharmProjects/Kaggle/DSB_data/stage1_train'


class InferenceConfig(CellsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


inference_config = InferenceConfig()
inference_config.display()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# Validation dataset
dataset_val = CellsDataset()
# dataset_val.load_cells(IMAGES_DIR, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.load_cells(IMAGES_DIR)
dataset_val.prepare()


for im_id in dataset_val.image_ids[9:10]:
    lm = dataset_val.load_mask(im_id)
    print(lm[0].shape)
    m = np.sum(lm[0], -1)
    print(m.shape)
    skimage.io.imsave('pred_mask2.png', m)
    print(dataset_val.image_info[im_id])
    original_image = dataset_val.load_image(im_id)

    results = model.detect([original_image], verbose=1)

    r = results[0]


    im = visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                     dataset_val.class_names, r['scores'], figsize=(18, 18), return_image=True)
    skimage.io.imsave('pred_im_01_2.png', im)
    # skimage.io.imsave('/home/futura/PycharmProjects/Kaggle/Data_Science_Bowl_2018_Mask_RCNN/test_img/orig_mask.png', mask)
