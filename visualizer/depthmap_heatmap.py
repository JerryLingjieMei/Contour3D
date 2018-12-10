from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
depthmap = np.load("../data/contour/depthmaps/00009.npy")
plt.matshow(np.transpose(depthmap), origin = 'bottom')
plt.show()