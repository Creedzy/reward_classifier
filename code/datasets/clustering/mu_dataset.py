from torch.utils import data
from PIL import Image
import cv2
import numpy
import os
from os import walk

IMAGE_PATH = "./code/datasets/test/"


class MU(data.Dataset):

    splits = ('train', 'train+unlabeled', 'unlabeled', 'test')

    def __init__(self, root, split='train', transform=None,
                 target_transform=None):
        super(MU, self).__init__()
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.labels = None
        np_array = []

        for (dirpath, dirnames, filenames) in walk(IMAGE_PATH):
            if filenames is None or len(filenames) is 0:
                exit("Naaaah, image sources missing")
            for filename in filenames:
                im_grey = cv2.imread(IMAGE_PATH + filename,
                                     cv2.IMREAD_COLOR)
                resized = cv2.resize(im_grey, (256, 256))
                im = Image.open(IMAGE_PATH + filename)

                np_array.append(resized)

        np_array = numpy.array(np_array)
        self.data = numpy.reshape(np_array, [-1, 256, 256, 3])
        self.data = numpy.squeeze(np_array)
        print('images', self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            img, target = self.data[idx], int(self.labels[idx])
        else:
            img, target = self.data[idx], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(numpy.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _verify_folds(self, folds):
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = ("Value for argument folds should be in the range [0, 10), "
                   "but got {}.")
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))
