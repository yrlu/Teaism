
import sys
import numpy as np

# Set your own Caffe root
caffe_root = '/home/jyh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

net = caffe.Net("deploy.prototxt", "bvlc_reference_caffenet.caffemodel", caffe.TEST);

f_out = open('model.txt', 'w')

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
for l in layers:
  print l, "weights: ", net.params[l][0].data.shape
  params = net.params[l][0].data.flatten().tolist()
  for i in params:
    f_out.write(str(i) + " ")
  f_out.write("\n")

  print l, "biases: ", net.params[l][0].data.shape
  params = net.params[l][1].data.flatten().tolist()
  for i in params:
    f_out.write(str(i) + " ")
  f_out.write("\n")

f_out.close()
