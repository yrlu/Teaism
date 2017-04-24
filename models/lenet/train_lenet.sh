#!/usr/bin/env sh
set -e

/home/jyh/caffe/build/tools/caffe train --solver=lenet_solver.prototxt &> log
