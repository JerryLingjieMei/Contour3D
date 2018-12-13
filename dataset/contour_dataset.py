from util.utils import *
import numpy as np
from imageio import imread
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

from util.utils import DATA_FOLDER, CONTOUR_CONTOUR_FOLDER, CONTOUR_DEPTHMAP_FOLDER
from dataset.base_dataset import BaseDataset


class ContourDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.is_train = opt.isTrain

    def __init__(self):
        self.is_train = None
        self.contour_folder = CONTOUR_CONTOUR_FOLDER
        self.data_folder = DATA_FOLDER
        self.train_length = 20000
        self.test_length = 2000
        self.cycle = 0

    def __len__(self):
        if self.is_train:
            return self.train_length
        else:
            return self.test_length

    def transform(self, contour, depthmap):
        if self.is_train:
            i, j = np.random.randint(0, 128), np.random.randint(0, 128)
            depthmap = depthmap[i:i + 384, j:j + 384, :]
            contour = contour[i:i + 384, j:j + 384, :]
        else:
            i = self.cycle // 2 * 128
            j = self.cycle % 2 * 128
            contour = contour[i:i + 384, j:j + 384, :]
            depthmap = depthmap[i:i + 384, j:j + 384, :]

        contour = TF.to_tensor(contour)
        depthmap = TF.to_tensor(depthmap)
        contour = TF.normalize(contour, mean=[.5, .5, .5], std=[.5, .5, .5])
        depthmap = TF.normalize(depthmap, mean=[.5], std=[.5])
        return contour, depthmap

    def __getitem__(self, item):
        if self.is_train:
            depth_path = os.path.join(CONTOUR_DEPTHMAP_FOLDER, "{:05d}.npy".format(item))
            # heightmap = np.load(os.path.join(HEIGHTMAP_FOLDER, "{:05d}.png".format(item)))
            contour_path = os.path.join(CONTOUR_CONTOUR_FOLDER, "{:05d}.png".format(item))
        else:
            depth_path = os.path.join(CONTOUR_DEPTHMAP_FOLDER, "{:05d}.npy".format(item + self.train_length))
            # heightmap = np.load(os.path.join(HEIGHTMAP_FOLDER, "{:05d}.png".format(item + self.train_length)))
            contour_path = os.path.join(CONTOUR_CONTOUR_FOLDER, "{:05d}.png".format(item + self.train_length))
        contour = imread(contour_path)
        depthmap = np.load(depth_path)
        depthmap = np.rot90(depthmap) / 500.
        depthmap = np.expand_dims(depthmap, 2)
        depthmap = np.array(depthmap, dtype=np.float)
        contour = np.array(contour[3:515, 3:515, 0:3], dtype=np.float)
        xs, ys = np.meshgrid(np.arange(0, 1, 1 / contour.shape[0]), np.arange(0, 1, 1 / contour.shape[1]))
        contour[:, :, 0] = contour[:, :, 0] / 256
        contour[:, :, 1] = xs
        contour[:, :, 2] = ys
        contour, depthmap = self.transform(contour, depthmap)
        contour = contour.float()
        depthmap = depthmap.float()
        if not self.is_train:
            self.cycle = (self.cycle + 1) % 4
        return dict(A=contour, B=depthmap, A_paths=contour_path, B_paths=depth_path)

    def name(self):
        return 'ContourDataset'
