
import numpy as np
import scipy.misc as spmi

img = spmi.imread("cat-01.jpg")
img = spmi.imresize(img, [227, 227])

spmi.imsave("cat.bmp", img)
