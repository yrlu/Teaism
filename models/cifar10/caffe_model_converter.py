
import sys
import numpy as np

# Set your own Caffe root
caffe_root = '/home/jyh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

caffe.set_mode_cpu()
net = caffe.Net("test.prototxt", "snapshots/cifar10_iter_1000.caffemodel", caffe.TEST);

f_out = open('model.txt', 'w')

layers = ['conv1','conv2','conv3', 'fc4', 'fc5']
for l in layers:
  print l, "weights: ", net.params[l][0].data.shape
  params = net.params[l][0].data.flatten().tolist()
  for cnt, p in enumerate(params):
    f_out.write(str(p) + " ")
#    print cnt, p
  f_out.write("\n")

  print l, "biases: ", net.params[l][1].data.shape
  params = net.params[l][1].data.flatten().tolist()
  for i in params:
    f_out.write(str(i) + " ")
  f_out.write("\n")


f_out.close()
