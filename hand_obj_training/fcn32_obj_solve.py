caffe_root = '/home/minghuam/caffe-fcn/'
import sys,os
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import argparse
import numpy as np

# make a bilinear interpolation kernel
# credit @longjon
def upsample_filt(size):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)

# set parameters s.t. deconvolutional layers compute bilinear interpolation
# N.B. this is for deconvolution without groups
def interp_surgery(net, layers):
    for l in layers:
        print net.params[l][0].data.shape
        m, k, h, w = net.params[l][0].data.shape
        print m,k,h,w
        if m != k:
            print 'input + output channels need to be the same'
            raise
        if h != w:
            print 'filters need to be square'
            raise
        filt = upsample_filt(h)
        net.params[l][0].data[range(m), range(k), :, :] = filt
        # for i in range(m):
        #     for j in range(k):
        #         net.params[l][0].data[i, j, :, :] = filt


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

parser = argparse.ArgumentParser()
parser.add_argument('base_weights', help = 'Base weights')
parser.add_argument('source', help = 'Source txt file')
parser.add_argument('model_prefix', help = 'Model file prefix')
parser.add_argument('max_iterations', help = 'Max training iterations')
parser.add_argument('cpu_id', help = 'CPU ID')
args = parser.parse_args()

train_prototxt = 'fcn32_obj_train.prototxt'
solver_prototxt = 'fcn32_obj_solver.prototxt'
new_train_prototxt = set_properties(train_prototxt, {'source' : args.source})
new_solver_prototxt = set_properties(solver_prototxt, 
    {
        'net': new_train_prototxt,
        'snapshot_prefix': args.model_prefix
    })

# init
caffe.set_mode_gpu()
caffe.set_device(int(args.cpu_id))

solver = caffe.SGDSolver(new_solver_prototxt)

# do net surgery to set the deconvolution weights for bilinear interpolation
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
interp_surgery(solver.net, interp_layers)

# copy base weights for fine-tuning
solver.net.copy_from(args.base_weights)

# solve straight through -- a better approach is to define a solving loop to
# 1. take SGD steps
# 2. score the model by the test net `solver.test_nets[0]`
# 3. repeat until satisfied
#solver.step(4000)
solver.step(int(args.max_iterations))