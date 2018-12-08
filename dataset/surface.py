import math

import numpy as np

from dataset.utils import *


class Surface:
    def __init__(self, shapes, size):
        self.size = size
        self.basics = shapes

        self.x, self.y = np.meshgrid(np.arange(0, size[0]), np.arange(0, size[1]))
        self.z = np.zeros(size)
        for f in self.basics:
            self.z += f(self.x, self.y)

    def get_height(self, x, y):
        h = 0
        for f in self.basics:
            h += f(x, y)
        return h

    def normal_vector(self, x, y, z):
        shape = x.shape
        return np.cross(np.stack((np.full(shape, DELTA), np.zeros(shape), self.get_height(x + DELTA, y) - z), axis=-1),
                        np.stack((np.zeros(shape), np.full(shape, DELTA), self.get_height(x, y + DELTA) - z), axis=-1))

    def get_surface(self):
        return self.z

    def get_contours(self, n_contours=10, n_samples=1000, sight_angle=45):
        perspective = normalize(np.array([1, 1, -math.tan(sight_angle / 180 * math.pi)]))
        px = normalize(np.cross(perspective, np.array([0, 0, 1])))
        py = normalize(np.cross(px, perspective))
        res = []
        now = []

        n_buckets = n_samples * 10
        buckets = [-1e9] * (n_buckets + 1)
        for k in range(n_contours):
            if k > 0:
                n_in_gap = max(1, 1000 // n_contours)
                for i in range(n_in_gap):
                    xs = np.full((n_samples,),
                                 SURFACE_SIZE / (n_contours - 1) * (k - 1) + SURFACE_SIZE / (n_contours - 1) / (
                                         n_in_gap + 1) * (i + 1))
                    ys = np.arange(0, n_samples, 1) * SURFACE_SIZE / (n_samples - 1)
                    zs = self.get_height(xs, ys)
                    bids = np.round((xs - ys + SURFACE_SIZE) / (2 * SURFACE_SIZE) * n_buckets).astype(int)
                    for x, y, z, bid in zip(xs, ys, zs, bids):
                        p = np.array([x, y, z])
                        buckets[bid] = max(buckets[bid], float(np.inner(p, py)))
            xs = np.full((n_samples,), SURFACE_SIZE / (n_contours - 1) * k)
            ys = np.arange(0, n_samples, 1) * (SURFACE_SIZE / (n_samples - 1))
            zs = self.get_height(xs, ys)
            nvs = self.normal_vector(xs, ys, zs)
            bids = np.round((xs - ys + SURFACE_SIZE) / (2 * SURFACE_SIZE) * n_buckets).astype(int)
            for x, y, z, nv, bid in zip(xs, ys, zs, nvs, bids):
                p = np.array([x, y, z])
                if np.inner(perspective, nv) < 0 and np.inner(p, py) > buckets[bid]:
                    now.append((np.inner(p, px), np.inner(p, py)))
                    buckets[bid] = float(np.inner(p, py))
                elif len(now) != 0:
                    res.append(now)
                    now = []
            if len(now) != 0:
                res.append(now)
                now = []
        return res
