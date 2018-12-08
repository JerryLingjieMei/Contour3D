import torch
import torchvision
from torch.utils.data import Dataset
from imageio import imread
from lib.utils import *
import numpy as np


class Contours(Dataset):

    def __init__(self, phase):
        super(self, _).__init__()
        self.phase = phase
        self.contour_folder = CONTOUR_FOLDER
        self.data_folder = DATA_FOLDER
        self.train_length = 10000
        self.test_length = 2000

    def __len__(self):
        if self.phase == "train":
            return self.train_length
        else:
            return self.test_length

    def __getitem__(self, item):
        if self.phase == "train":
            heightmap = np.load(os.path.join(HEIGHTMAP_FOLDER, "{:05d}.png".format(item)))
            contour = imread(os.path.join(CONTOUR_FOLDER, "{:05d}.png".format(item)))[:, :, 0]
        else:
            heightmap = np.load(os.path.join(HEIGHTMAP_FOLDER, "{:05d}.png".format(item + self.train_length)))
            contour = imread(os.path.join(CONTOUR_FOLDER, "{:05d}.png".format(item + self.test_length)))[:, :, 0]
        return heightmap, contour
