
import numpy as np
import scipy.misc as scimi

with open('train.txt') as f:
  lines = f.readlines()

for line in lines[0:10]:
  filename = line.split()[0]
  print filename
  img = scimi.imread(filename)
  new_img = np.empty([28, 28, 3])
  new_img[:,:,0] = img
  new_img[:,:,1] = img
  new_img[:,:,2] = img
  print img
  scimi.imsave(filename, img)

