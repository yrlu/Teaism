
import numpy as np
import scipy.misc as spmi
import sys
caffe_root = '/home/jyh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net = caffe.Net("deploy.prototxt", "bvlc_reference_caffenet.caffemodel", caffe.TEST)

img = spmi.imread("cat.bmp").astype(dtype=np.float32)
img -= [128, 128, 128]
img = img.transpose([2,0,1])

net.blobs['data'].data[...] = img

net.forward()

print net.blobs['prob'].data[...]
