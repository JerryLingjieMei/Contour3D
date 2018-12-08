import numpy as np


class SphericalCap:
    def __init__(self, a, x0, y0, r):
        if a > r:
            raise ValueError
        self.a = a
        self.x0 = x0
        self.y0 = y0
        self.r = r

    def __call__(self, x, y):
        t = self.r ** 2 - (x - self.x0) ** 2 - (y - self.y0) ** 2
        t = np.clip(t, 0, None)
        return np.clip(t ** .5 - self.r + self.a, None, 0)


class SinusoidWaves:
    def __init__(self, a, x0, y0):
        self.a = a
        self.x0 = x0
        self.y0 = y0
        self.lambda_squared = x0 ** 2 + y0 ** 2

    def __call__(self, x, y):
        return self.a * np.sin((x * self.x0 + y * self.y0) / self.lambda_squared * 2 * np.pi)


class Rings:
    def __init__(self, a, x0, y0, r, sigma):
        self.A = a
        self.x0 = x0
        self.y0 = y0
        self.r = r
        self.sigma = sigma

    def __call__(self, x, y):
        d = np.sqrt((x - self.x0) ** 2 + (y - self.y0) ** 2)
        return self.A * np.exp(-(d - self.r) ** 2 / (2 * (self.sigma ** 2)))
