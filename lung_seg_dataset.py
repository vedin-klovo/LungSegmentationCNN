import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as Transforms
import os
import logging
from PIL import Image
import numpy as np

from typing import Tuple


class LungSegmentationDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            mask_suffix: str,
            transforms: Transforms,
            input_size: Tuple[int, int],
            output_size: Tuple[int, int]
    ):
        self.root_dir = root_dir
        self.transforms = transforms
        self.mask_suffix = mask_suffix
        self.input_size = input_size
        self.output_size = output_size

        # data loading
        self.images_dir = os.path.join(root_dir, 'images')
        self.masks_dir = os.path.join(root_dir, 'masks')
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.images_dir) if not file.startswith('.')]
        self.ids = [i for i in self.ids if i + self.mask_suffix + '.png' in os.listdir(self.masks_dir)]
        self.ids = self.ids[:637]
        logging.info(f"Created a lung segmentatation dataset with {len(self.ids)} samples.")

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(image: Image, new_size: Tuple[int, int], is_mask: bool = False):
        image = image.resize(new_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
        image_numpy = np.asarray(image)

        if not is_mask:
            if image_numpy.ndim == 2:
                image_numpy = image_numpy[np.newaxis, ...]
            else:
                image_numpy = image_numpy.transpose((2, 0, 1))
            image_numpy = image_numpy / 255.0
            return image_numpy
        else:
            labels = np.unique(image_numpy)
            image_numpy_fin = np.zeros(image_numpy.shape)
            for i, val in enumerate(list(labels)):
                image_numpy_fin[image_numpy == val] = i
            return image_numpy_fin

    def __getitem__(self, idx):
        name = self.ids[idx]
        image_file_path = os.path.join(self.images_dir, name + '.png')
        mask_file_path = os.path.join(self.masks_dir, name + self.mask_suffix + '.png')

        image = Image.open(image_file_path)
        mask = Image.open(mask_file_path)

        assert image.size == mask.size, \
            f"Image and mask {name} should be the same size, {image.size} and  {mask.size} provided instead"

        image = self.preprocess(image, self.input_size, is_mask=False)
        mask = self.preprocess(mask, self.output_size, is_mask=True)

        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class ShenzhenDataset(LungSegmentationDataset):
    def __init__(
            self,
            root_dir: str = 'data/Shenzhen',
            mask_suffix: str = '_mask',
            transforms: Transforms = None,
            input_size: Tuple[int, int] = (572, 572),
            output_size: Tuple[int, int] = (388, 388)
    ):
        super().__init__(root_dir, mask_suffix, transforms, input_size, output_size)


class MontgomeryDataset(LungSegmentationDataset):
    def __init__(
            self,
            root_dir: str = 'data/Montgomery',
            mask_suffix: str = '',
            transforms: Transforms = None,
            input_size: Tuple[int, int] = (572, 572),
            output_size: Tuple[int, int] = (388, 388)
    ):
        super().__init__(root_dir, mask_suffix, transforms, input_size, output_size)


class JSRTDataset(LungSegmentationDataset):
    def __init__(
            self,
            root_dir: str = 'data/JSRT',
            mask_suffix: str = '_label',
            transforms: Transforms = None,
            input_size: Tuple[int, int] = (256, 256),
            output_size: Tuple[int, int] = (256, 256),
    ):
        super(JSRTDataset, self).__init__(root_dir, mask_suffix, transforms, input_size, output_size)


if __name__ == '__main__':
    shenzhen = ShenzhenDataset()
    print(len(shenzhen))
    montgomery = MontgomeryDataset()
    print(len(shenzhen))
    jsrt = JSRTDataset()
    print(len(jsrt))


