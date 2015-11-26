import os,sys,cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

caffe_root = '/home/minghuam/caffe-fcn'
sys.path.insert(0, caffe_root + '/python')
import caffe

deploy_file = '/home/minghuam/egocentric_action_recognition/hand/fcn32_obj/fcn32_obj_deploy.prototxt'
model_file = '/home/minghuam/egocentric_action_recognition/hand/fcn32_obj/GTEA_model/FCN32_OBJ_GTEA_S1_iter_4000.caffemodel'
test_file = '/home/minghuam/egocentric_action_recognition/hand/fcn32_obj/GTEA_data/s1_train.txt'
root_dir = '/home/minghuam/egocentric_action_recognition/hand/fcn32_obj'
mean_values = (104.00699, 116.66877, 122.67892)

Imean = np.ones((224, 224, 3))
for c in range(3):
    Imean[...,c] *= mean_values[c]


max_activations_dir = 'gtea_fcn32'
# max_images_dir = 'gtea_imgs'

# deploy_file = '/home/minghuam/egocentric_action_recognition/verb/verb_deploy_plus44.prototxt'
# model_file = '/home/minghuam/egocentric_action_recognition/verb/PLUS44_model/VERB_PLUS44_AHMAD_iter_3000.caffemodel'
# test_file = '/home/minghuam/egocentric_action_recognition/verb/PLUS44_data/ahmad_train.txt'
# images_dir = '/home/minghuam/data/GTEA_gaze_plus/RAW/images44/'
# max_activations_dir = 'plus44'
# max_images_dir = 'plus44_imgs'

def ls_images(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d)])

# flow_folders = []
# with open(test_file, 'r') as fr:
#     for line in fr.readlines():
#         tokens = line.strip().split(' ')
#         flow_folders.append((tokens[0], int(tokens[1])))

# max_activations = dict()
# for d in os.listdir(max_activations_dir):
#     if not os.path.isdir(os.path.join(max_activations_dir, d)):
#         continue
#     max_activations[d] = ls_images(os.path.join(max_activations_dir, d))

# max_images = dict()
# for d in os.listdir(max_images_dir):
#     if not os.path.isdir(os.path.join(max_images_dir, d)):
#         continue
#     max_images[d] = ls_images(os.path.join(max_images_dir, d))


# Iwheel = cv2.imread(os.path.join(max_activations_dir, 'color_wheel.jpg'))
# cv2.imshow('wheel', Iwheel)

image_pairs = []
with open(test_file, 'r') as fr:
    for line in fr.readlines():
        tokens = line.strip().split(' ')
        raw_img = os.path.join(root_dir, tokens[0])
        hand_img = os.path.join(root_dir, tokens[1])
        obj_img = os.path.join(root_dir, tokens[2])
        image_pairs.append((raw_img, hand_img, obj_img))

# max_activations = dict()
# for d in os.listdir(max_activations_dir):
#     if not os.path.isdir(os.path.join(max_activations_dir, d)):
#         continue
#     max_activations[d] = ls_images(os.path.join(max_activations_dir, d))

caffe.set_mode_gpu()
caffe.set_device(1)
net = caffe.Net(deploy_file, model_file, caffe.TEST)

def viz_images(Iimgs):
    h,w = Iimgs[0].shape[:2]
    Iret = np.zeros((h, w*len(Is), 3), np.uint8)
    for index,I in enumerate(Iimgs):
        Iret[:,index*w:index*w+w,:] = I
    return Iret

def viz_flow(Is):
    h,w = Is[0][0].shape[:2]
    Iret = np.zeros((h, w*len(Is), 3), np.uint8)
    for index,(Ix, Iy) in enumerate(Is):
        # Ix = Ix - 128.0
        # Iy = Iy - 128.0
        angles = np.arctan2(Iy, Ix)
        dists = np.sqrt(np.power(Ix, 2), np.power(Iy, 2))
        dists *= 1
        Hmax = 179
        H = ((angles + np.pi)/(2*np.pi)*Hmax).astype(np.uint8)
        Dmax = np.sqrt(127*127 + 127*127)
        S = (np.clip(dists/Dmax, 0, 1)*255).astype(np.uint8)
        HSV = np.ones((Ix.shape[0], Ix.shape[1], 3), np.uint8)*255
        HSV[...,0] = H
        HSV[...,1] = S
        Iret[:,index*w:index*w+w,:] = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
    return Iret

def viz_layer(net, layer, Iraw, filter_index):
    '''C x H x W'''
    data = net.blobs[layer].data[0,...]
    #print data.shape
    #print data.min(), data.max()
    r = data.max() - data.min()
    data = (data - data.min())/r
    res_data = (data*255).astype(np.uint8)
    c,h,w = res_data.shape[:]
    n = int(np.ceil(np.sqrt(c)))
    pad_size = ((0, n*n - c), (1, 1), (1, 1))
    pad_values = ((0,0), (255, 255), (255, 255))
    data = np.pad(res_data, pad_size, mode = 'constant', constant_values = pad_values)
    c,h,w = data.shape[:]
    I = data.reshape((h, w*n*n))
    data = data.reshape((n,n,h,w))
    data = data.transpose((0, 2, 1, 3))
    Ires = data.reshape((h*n, w*n))
    Ires = cv2.resize(Ires, (800, 800))
    Ires = cv2.cvtColor(Ires, cv2.COLOR_GRAY2BGR)

    key = 0
    # filter_index = 0
    n_filters = net.blobs[layer].data.shape[1]
    while key != ord('q'):
        #print 'filter:',filter_index
        Idraw = Ires.copy()
        size = Ires.shape[0]/n
        y = size*(filter_index/n)
        x = size*(filter_index%n)
        cv2.rectangle(Idraw, (x,y), (x+size, y+size), (0,0,255), 2)
        cv2.imshow('response', Idraw)
        # Ia = cv2.imread(max_activations[layer][filter_index])
        # cv2.imshow('max_activations', Ia)
        # Ib = cv2.imread(max_images[layer][filter_index])
        # cv2.imshow('raw_images', Ib)
        Iunit = cv2.resize(res_data[filter_index,...], (224, 224), interpolation = cv2.INTER_NEAREST)
        cv2.imshow('filter response', Iunit)

        # if Iflow.shape[1] != Ia.shape[1]:
        #     w = Ia.shape[1]
        #     h = w*Iflow.shape[0]/Iflow.shape[1]
        #     Iflow = cv2.resize(Iflow, (w,h))
        # cv2.imshow('flow', Iflow)
        # if Iimg.shape[1] != Ia.shape[1]:
        #     w = Ia.shape[1]
        #     h = w*Iimg.shape[0]/Iimg.shape[1]
        #     Iimg = cv2.resize(Iimg, (w,h))
        cv2.imshow('image', Iraw)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            sys.exit(0)
        new_index = filter_index
        if key == ord('l'):
            new_index = filter_index + 1
        if key == ord('j'):
            new_index = filter_index - 1
        if key == ord('i'):
            new_index = filter_index - n
        if key == ord('k'):
            new_index = filter_index + n
        if new_index >= 0 and new_index < n_filters:
            filter_index = new_index
            print 'filter:',filter_index

        if key == ord('s'):
            print 'filter', filter_index
            cv2.imwrite('Ia.jpg', Ia)
            cv2.imwrite('Iraw.jpg', Iraw)
            cv2.imwrite('Iunit.jpg', Iunit)

    return filter_index

crop_size = 224
filter_index = 0
for (raw_img, hand_img, obj_img) in image_pairs[:]:
    print raw_img, hand_img, obj_img
    I = cv2.imread(raw_img)
    x = (I.shape[1] - crop_size)/2
    y = (I.shape[0] - crop_size)/2
    I = I[y:y+crop_size, x:x+crop_size, :]
    Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
    Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
    net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
    out = net.forward()
    filter_index = viz_layer(net, 'conv5_3', I, filter_index)