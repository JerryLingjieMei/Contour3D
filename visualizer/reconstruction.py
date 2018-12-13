import numpy as np
from dataset.utils import *
import matplotlib.pyplot as plt
import torch.nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ReconstructionLoss(torch.nn.Module):
    def __init__(self, sight_angle):
        super().__init__()
        self.perspective = normalize(np.array([1, 1, -math.tan(sight_angle / 180 * math.pi)]))
        self.px = normalize(np.cross(self.perspective, np.array([0, 0, 1])))
        self.py = normalize(np.cross(self.px, self.perspective))

        self.xl = np.inner(np.array([0, SURFACE_SIZE, 0]), self.px)
        self.xr = np.inner(np.array([SURFACE_SIZE, 0, 0]), self.px)
        self.yl = np.inner(np.array([0, 0, 0]), self.py)
        self.yr = np.inner(np.array([SURFACE_SIZE, SURFACE_SIZE, 0]), self.py)

        gx = .5 / SURFACE_SIZE * (self.xr - self.xl) + self.xl
        gy = .5 / SURFACE_SIZE * (self.yr - self.yl) + self.yl
        self.base = torch.Tensor(self.px * gx + self.py * gy)
        self.stride = SURFACE_SIZE / (20 - 1)

    def reconstruct_3d(self, depth_map, target):
        depth_map = torch.rot90(depth_map, dims=(2, 3))
        x_map = self.perspective[0] * depth_map[:, 0, :, :] + self.px[0] * (self.xr - self.xl) * target[:, 1, :, :] \
                + self.py[0] * (self.yr - self.yl) * target[:, 2, :, :] + self.base[0]
        return torch.rot90(x_map, dims=(1, 2))

    def plot_contour(self, coord_map):
        """
        :param coord_map: b x 3 x x x y
        """
        return 1 - torch.exp(-self.stride ** 2 * (1 - torch.cos(coord_map * (np.pi / self.stride)) ** 2))

    def forward(self, input, target):
        input = input * .5 + .5
        input = input * 500
        target = target * .5 + .5
        reconstructed_3d = self.reconstruct_3d(input, target)
        contour = self.plot_contour(reconstructed_3d)
        contour_gt = target[:, 0, :, :]
        return F.l1_loss(contour, contour_gt)
