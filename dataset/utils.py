import numpy as np
import os

SURFACE_SIZE = 256
DELTA = .01

SINUSOID_R_MIN = 40
SINUSOID_R_MAX = 100
SINUSOID_A_MIN = 5
SINUSOID_A_MAX = 20
SINUSOID_THETA_THRESH = .2

SPHERICAL_R_MIN = 20
SPHERICAL_R_MAX = 60

RING_R_MIN = 30
RING_R_MAX = 120
RING_RATIO_MIN = .3
RING_RATIO_MAX = .15
RING_SIGMA_MIN = .2
RING_SIGMA_MAX = .8


def normalize(x):
    return x / np.linalg.norm(x)


def clip_int(x):
    return int(min(x, SURFACE_SIZE - 1))
