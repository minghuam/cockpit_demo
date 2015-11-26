import os,sys,cv2
import numpy as np
import heapq
from collections import deque
import matplotlib.pyplot as plt
import pickle
import copy

caffe_root = '/home/minghuam/caffe-dev'
sys.path.insert(0, caffe_root + '/python')
import caffe

# deploy_file = '/home/minghuam/egocentric_action_recognition/verb/verb_deploy_gtea.prototxt'
# model_file = '/home/minghuam/egocentric_action_recognition/verb/GTEA_model/VERB_GTEA_S1_iter_2000.caffemodel'
# test_file = '/home/minghuam/egocentric_action_recognition/verb/GTEA_data/S1_train.txt'

deploy_file = '/home/minghuam/egocentric_action_recognition/verb/verb_deploy_plus44.prototxt'
model_file = '/home/minghuam/egocentric_action_recognition/verb/PLUS44_model/VERB_PLUS44_AHMAD_iter_3000.caffemodel'
test_file = '/home/minghuam/egocentric_action_recognition/verb/PLUS44_data/ahmad_train.txt'


flow_folders = []
with open(test_file, 'r') as fr:
    for line in fr.readlines():
        tokens = line.strip().split(' ')
        flow_folders.append((tokens[0], int(tokens[1])))

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(deploy_file, model_file, caffe.TEST)

top_number = 9
max_activations = dict()
for layer in net.blobs.keys():
    data = net.blobs[layer].data[0,...]
    print layer, data.shape
    c = data.shape[0]
    max_activations[layer] = list()
    for i in range(c):
        top_list = list()
        for j in range(top_number):
            top_list.append((-float('inf'),'None','None'))
        max_activations[layer].append(top_list)

def viz(data):
    '''C x H x W'''
    print data.min(), data.max()
    r = data.max() - data.min()
    data = (data - data.min())/r
    data = (data*255).astype(np.uint8)
    c,h,w = data.shape[:]
    n = int(np.ceil(np.sqrt(c)))
    pad_size = ((0, n*n - c), (1, 1), (1, 1))
    pad_values = ((0,0), (255, 255), (255, 255))
    data = np.pad(data, pad_size, mode = 'constant', constant_values = pad_values)
    #return data[0,...]
    c,h,w = data.shape[:]
    # data = data.transpose((1,0,2))
    I = data.reshape((h, w*n*n))
    data = data.reshape((n,n,h,w))
    data = data.transpose((0, 2, 1, 3))
    I = data.reshape((h*n, w*n))
    I = cv2.resize(I, (1624, 1624))
    return I

for (flow_folder,label) in flow_folders[0:]:
    print flow_folder
    x_folder = os.path.join(flow_folder, 'x')
    y_folder = os.path.join(flow_folder, 'y')
    x_images = sorted([os.path.join(x_folder, f) for f in os.listdir(x_folder)])
    y_images = sorted([os.path.join(y_folder, f) for f in os.listdir(y_folder)])

    num_stack_frames = 10
    crop_size = 224
    mean = 128.0
    Is = deque()
    for index, (x_img, y_img) in enumerate(zip(x_images, y_images)):
        Ix = cv2.imread(x_img)
        Iy = cv2.imread(y_img)
        x = (Ix.shape[1] - crop_size)/2
        y = (Ix.shape[0] - crop_size)/2
        Ix = Ix[y:y+crop_size, x:x+crop_size, 0].astype(np.float64) - mean
        Iy = Iy[y:y+crop_size, y:y+crop_size, 0].astype(np.float64) - mean
        Is.append((Ix, Iy))

        if len(Is) == num_stack_frames:
            input_data = np.zeros((1, num_stack_frames*2, crop_size, crop_size), np.float64)
            for i, (Ix, Iy) in enumerate(Is):
                input_data[0,i*2+0,...] = Ix
                input_data[0,i*2+1,...] = Iy

            net.blobs['data'].data[...] = input_data
            net.forward()
            score = net.blobs['score'].data.ravel()
            print label, np.argmax(score)

            for layer in max_activations:
                # layer = 'conv1'
                folder = os.path.dirname(os.path.dirname(x_img))
                basename = os.path.basename(x_images[index - num_stack_frames + 1])
                data = net.blobs[layer].data[0,...]
                if len(data.shape) > 1:
                    activations = np.sum(data, axis = (1,2))
                else:
                    activations = data

                # print layer, activations.shape
                for i in range(activations.shape[0]):
                    entry = (activations[i], folder, basename)
                    h = max_activations[layer][i]
                    heapq.heappush(h, entry)
                    heapq.heappop(h)
                    max_activations[layer][i] = h

            # print max_activations['fc7']
            # output_data = net.blobs['conv5'].data[0,...]
            # Iviz = viz(output_data)
            # cv2.imshow('Iresponse', Iviz)
            # cv2.waitKey(0)
            # Is.popleft()

#save_file = os.path.basename(test_file).splitext()[0] + '.pkl'
save_file = 'plus44.pkl'
pickle.dump(max_activations, open(save_file, 'w'))


