import os
import glob

import paddle

from transforms import Compose

class Dataset(paddle.io.Dataset):
    def __init__(self, dataset_root=None, transforms=None):
        if dataset_root is None:
            raise ValueError("dataset_root is None")
        self.dataset_root = dataset_root
        if transforms is not None:
            self.transforms = Compose(transforms)
        else:
            self.transforms = None

        self.im1_list = glob.glob(os.path.join(self.dataset_root, "images", "*.jpg"))
        self.im2_list = glob.glob(os.path.join(self.dataset_root, "gts", "*.jpg"))

        self.im1_list.sort()
        self.im2_list.sort()
        assert len(self.im1_list) == len(self.im2_list)

    def __getitem__(self, index):
        im1 = self.im1_list[index]
        im2 = self.im2_list[index]
        if self.transforms is not None:
            return self.transforms(im1, im2)
        else:
            return im1, im2

    def __len__(self):
        return len(self.im1_list)

if __name__ == '__main__':
    dataset = Dataset(dataset_root="/Users/alex/Downloads/moire_competition_dataset_1206/moire_train_dataset")

    for d in dataset:
        pass




