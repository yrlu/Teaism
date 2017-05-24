
import numpy as np
import scipy.misc as scimi

with open('test.txt') as f:
  lines = f.readlines()

with open('test.txt', 'w') as f:
  for line in lines:
    filename = line.split()[0][:-4]
    print filename
    f.write('datasets/cifar10/bmp_imgs/' + filename + '.bmp ' + line.split()[1] + '\n')

