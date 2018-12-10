import numpy as np
import os
import math

SURFACE_SIZE = 256
DEPTHMAP_SIZE = 512
BUCKET_SIZE = 512
DELTA = .01
DEFAULT_SIGHT_ANGLE = 53

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

def to_index_range(x, n):
    return max(0,min(n-1,math.floor(x*n)))

def margin_cut_function(x):
    margin = SURFACE_SIZE/4
    alpha = np.minimum(np.maximum(np.minimum(x, SURFACE_SIZE-x), 0), margin)
    theta = (alpha/margin-.5)*math.pi
    #return np.zeros(x.shape)
    return (np.sin(theta)+1)/2