import math
import random
import matplotlib.pyplot as plt

from dataset.shapes import SphericalCap, SinusoidWaves, Rings
from dataset.surface import Surface
from dataset.utils import *


def sinusoid_wave():
    r = np.random.uniform(SINUSOID_R_MIN, SINUSOID_R_MAX)
    a = np.random.uniform(SINUSOID_A_MIN, SINUSOID_A_MAX)
    theta = np.random.uniform(-np.pi, np.pi)
    return SinusoidWaves(a, r * np.cos(theta), r * np.sin(theta))


def spherical_cap():
    x0 = np.random.uniform(0, SURFACE_SIZE)
    y0 = np.random.uniform(0, SURFACE_SIZE)
    r = np.random.uniform(SPHERICAL_R_MIN, SPHERICAL_R_MAX)
    a = min(np.random.uniform(0, r), np.random.uniform(0, r))
    return SphericalCap(a, x0, y0, r)


def ring():
    x0 = random.uniform(0, SURFACE_SIZE)
    y0 = random.uniform(0, SURFACE_SIZE)
    r = random.uniform(RING_R_MIN, RING_R_MAX)
    a = random.uniform(r * RING_RATIO_MIN, r * RING_RATIO_MAX)
    sigma = random.uniform(a * RING_SIGMA_MIN, a * RING_SIGMA_MAX)
    return Rings(a, x0, y0, r, sigma)


def generate_surface():
    random.seed()
    n_shapes = random.randint(3, 6)
    basic_shapes = [sinusoid_wave, spherical_cap, spherical_cap, ring, ring, ring]
    return Surface([random.choice(basic_shapes)() for _ in range(n_shapes)], size=(SURFACE_SIZE, SURFACE_SIZE))
