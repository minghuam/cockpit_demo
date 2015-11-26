caffe_root = '/home/minghuam/caffe-dev/'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import argparse
import numpy as np

def set_properties(prototxt, properties):
    print properties
    basename = os.path.basename(prototxt)
    with open(prototxt, 'r') as fr:
        lines = fr.readlines()
    for i, line in enumerate(lines):
        for key in properties:
            if line.strip().startswith(key + ':'):
                index = line.index(':')
                if type(properties[key]) == str:
                    lines[i] = line[:index+1] + ''' "''' + properties[key] + '''"\n'''
                else:
                    lines[i] = line[:index+1] + ' ' + str(properties[key]) + '\n'
    new_prototxt = '.' + prototxt
    with open(new_prototxt, 'w') as fw:
        for line in lines:
            fw.write(line)
    return new_prototxt

base_weights = "VGG_CNN_M.caffemodel"
solver_prototxt = 'solver.prototxt'

# init
caffe.set_mode_gpu()
caffe.set_device(2)
solver = caffe.SGDSolver(solver_prototxt)
solver.net.copy_from(base_weights)
solver.step(4000)
