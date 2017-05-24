
import numpy as np
import scipy.misc as scimi

with open('test.txt') as f:
  lines = f.readlines()

for line in lines:
  filename = line.split()[0][:-4]
  print filename
  img = scimi.imread('imgs/' + filename + '.png')
#  print img
  scimi.imsave('bmp_imgs/' + filename + '.bmp', img)

