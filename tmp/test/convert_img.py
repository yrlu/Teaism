
import numpy as np
import scipy.misc as scimi

img = scimi.imread('steel_wool_large.bmp')
img = img[0:640, 0:640]
scimi.imsave('steel_wool_small.bmp', img)

img = scimi.imread('steel_wool_large_reference_output.bmp')
img = img[0:640, 0:640]
scimi.imsave('steel_wool_small_reference_output.bmp', img)
