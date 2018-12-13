import numpy as np
from dataset.utils import *
import matplotlib.pyplot as plt
import torch.nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from visualizer.reconstruction import ReconstructionLoss
from util.utils import *
import imageio
from dataset import CreateDataLoader
from options.train_options import TrainOptions

if __name__ == '__main__':
    opt = TrainOptions().parse()
    loader = CreateDataLoader(opt)
    loss = ReconstructionLoss(DEFAULT_SIGHT_ANGLE)
    dataloader = loader.load_data()
    for data in dataloader:
        loss.forward(data["B"], data["A"])
