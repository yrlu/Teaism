
import numpy as np
import scipy.misc as scimi

with open('train.txt') as f:
  lines = f.readlines()

for line in lines[0:10]:
  filename = line.split()[0][:-4]
  print filename
  img = scimi.imread('imgs/' + filename + '.png')
#  print img
  scimi.imsave('bmp_imgs/' + filename + '.bmp', img)

