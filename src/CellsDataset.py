from src.config import Config
import src.utils as utils
import os
import numpy as np
import skimage.io
import tifffile as tif
from skimage.transform import resize

class CellsConfig(Config):

    # Give the configuration a recognizable name
    NAME = "cell"
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


class CellsDataset(utils.Dataset):

    def load_cells(self, dataset_dir, height, width):
        self.add_class("cells", 1, "cell")
        image_ids = next(os.walk(dataset_dir))[1]

        for i in image_ids:
            self.add_image("cells", image_id=i,
                           width=width, height=height,
                           path=os.path.join(dataset_dir, i, 'images', i+'.png'),
                           path_mask=os.path.join(dataset_dir, i, 'masks'))

    def load_mask(self, image_id):

        instance_masks = []
        path_mask = self.image_info[image_id]['path_mask']
        masks_names = next(os.walk(path_mask))[2]
        height, width, instance_count = (self.image_info[image_id]['height'],
                                        self.image_info[image_id]['width'],
                                        len(masks_names))
        masks = np.zeros(shape=(height, width, instance_count))
        for i, mask in enumerate(masks_names):
            if mask.split('.')[-1] != 'png':
                continue
            img = skimage.io.imread(os.path.join(path_mask, mask))
            # img = resize(img, (height, width), mode='constant', preserve_range=True)
            instance_masks.append(img)

        masks = np.stack(instance_masks, axis=2)
        class_ids = np.ones(shape=(instance_count), dtype=np.int32)

        return masks, class_ids

