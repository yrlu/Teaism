
import numpy as np
import scipy.misc as scimi

img = scimi.imread('steel_wool_large.bmp')
img = img[0:227, 0:227]
scimi.imsave('steel_wool_median.bmp', img)
