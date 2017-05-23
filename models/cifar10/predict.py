
import numpy as np
import scipy.misc as spmi
import sys
caffe_root = '/home/jyh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net = caffe.Net("test.prototxt", "snapshots/cifar10_iter_20000.caffemodel", caffe.TEST)

for i in range(1):
  img = spmi.imread("../../datasets/cifar10/bmp_imgs/0000" + str(i+1) + ".bmp").astype(dtype=np.float32)
  img -= [128, 128, 128]
  img = img[:,:,::-1]
  img = img.transpose([2,0,1])
  label = np.empty([1,1,1,1])
  label[0] = 6

  net.blobs['data'].data[...] = img
  net.blobs['label'].data[...] = label

  net.forward()
  net.backward()

#  print net.blobs['prob'].data[...].tolist()
#  print net.blobs['prob'].data.shape
#  print net.params['conv1'][0].data
#  print net.params['conv1'][0].data.shape
  print net.blobs['fc5'].diff[...].tolist()
  print net.blobs['fc5'].diff.shape
  print net.params['fc5'][0].diff
  print net.params['fc5'][0].diff.shape

