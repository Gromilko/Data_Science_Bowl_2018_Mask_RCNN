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

IMAGES_DIR = r'/home/futura/PycharmProjects/Kaggle/DSB_data/stage1_test'

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


for im_id in dataset_val.image_ids:
    print(dataset_val.image_info[im_id])
    original_image = dataset_val.load_image(im_id)

    results = model.detect([original_image], verbose=1)

    r = results[0]

    # mask = 0
    # for i in range(gt_mask.shape[2]):
    #     mask += gt_mask[:, :, i]
    temp_dir = os.path.join('/home/futura/PycharmProjects/Kaggle/DSB_data/predict_stage1_test', dataset_val.image_info[im_id]['id'])
    os.mkdir(temp_dir)
    os.mkdir(os.path.join(temp_dir, 'images'))
    os.mkdir(os.path.join(temp_dir, 'masks'))

    skimage.io.imsave(os.path.join(temp_dir, 'images/image.png'), original_image)

    for i in range(r['masks'].shape[2]):
        print(i, end=' ')

        # m, bb, im_sh = r['masks'][:, :, i], r['rois'][i], original_image.shape
        # orig_mask = utils.unmold_mask(m, bb, image_meta[1:3])
        skimage.io.imsave(os.path.join(temp_dir, 'masks', '{}.png'.format(i)),
            np.where(r['masks'][:, :, i] > 0, 255, 0))
    print()

    # visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
    #                                  dataset_val.class_names, r['scores'], figsize=(18, 18), return_image=False)
    # skimage.io.imsave('/home/futura/PycharmProjects/Kaggle/Data_Science_Bowl_2018_Mask_RCNN/test_img/pred_mask.png', im)
    # skimage.io.imsave('/home/futura/PycharmProjects/Kaggle/Data_Science_Bowl_2018_Mask_RCNN/test_img/orig_mask.png', mask)
