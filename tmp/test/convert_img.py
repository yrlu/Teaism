
import numpy as np
import scipy.misc as scimi

img = scimi.imread('steel_wool_large.bmp')
img = img[0:28, 0:28]
scimi.imsave('steel_wool_small1.bmp', img)
scimi.imsave('steel_wool_small2.bmp', img)
