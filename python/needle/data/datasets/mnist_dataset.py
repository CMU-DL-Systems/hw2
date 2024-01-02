import gzip
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)

        with gzip.open(image_filename, 'rb') as image_file:
            image_data = image_file.read()
            num_images = int.from_bytes(image_data[4:8])
            imgs = np.frombuffer(image_data[16:], dtype=np.uint8)
            imgs = imgs.reshape(num_images, -1)
            imgs = imgs.astype(np.float32)
            min_val = imgs.min()
            max_val = imgs.max()
            imgs = (imgs - min_val) / (max_val - min_val)

        with gzip.open(label_filename, 'rb') as label_file:
            label_data = label_file.read()
            labels = np.frombuffer(label_data[8:], dtype=np.uint8)

        self.imgs = imgs
        self.labels = labels
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        imgs = self.imgs[index]
        origin_shape = imgs.shape
        imgs = super().apply_transforms(imgs.reshape((*origin_shape[:-1], 28, 28, 1))).reshape(origin_shape)
        labels = self.labels[index]
        return imgs, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.labels.shape[0]
        ### END YOUR SOLUTION