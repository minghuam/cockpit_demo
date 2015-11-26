import os,sys,cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import pickle

caffe_root = '/home/minghuam/egocentric_action_recognition/visualize/caffe-dev'
sys.path.insert(0, caffe_root + '/python')
import caffe

#deploy_file = '/home/minghuam/egocentric_action_recognition/verb/verb_deploy_gtea.prototxt'
#model_file = '/home/minghuam/egocentric_action_recognition/verb/GTEA_model/VERB_GTEA_S1_iter_2000.caffemodel'
#test_file = '/home/minghuam/egocentric_action_recognition/verb/GTEA_data/S1_train.txt'
#max_activations_dir = 'max_activations'

deploy_file = '/home/minghuam/egocentric_action_recognition/verb/verb_deploy_plus44.prototxt'
model_file = '/home/minghuam/egocentric_action_recognition/verb/PLUS44_model/VERB_PLUS44_AHMAD_iter_3000.caffemodel'
test_file = '/home/minghuam/egocentric_action_recognition/verb/PLUS44_data/ahmad_train.txt'
images_dir = '/home/minghuam/data/GTEA_gaze_plus/RAW/images44/'
max_activations_dir = 'plus44'
max_images_dir = 'plus44_imgs'

# max_activations_imgs = dict()
# max_dict = pickle.load(open('max.pkl'))
# for layer in max_dict:
#     print max_dict[layer]

# sys.exit(0)

def ls_images(d):
    return sorted([os.path.join(d, f) for f in os.listdir(d)])

flow_folders = []
with open(test_file, 'r') as fr:
    for line in fr.readlines():
        tokens = line.strip().split(' ')
        flow_folders.append((tokens[0], int(tokens[1])))

max_activations = dict()
for d in os.listdir(max_activations_dir):
    if not os.path.isdir(os.path.join(max_activations_dir, d)):
        continue
    max_activations[d] = ls_images(os.path.join(max_activations_dir, d))

max_images = dict()
for d in os.listdir(max_images_dir):
    if not os.path.isdir(os.path.join(max_images_dir, d)):
        continue
    max_images[d] = ls_images(os.path.join(max_images_dir, d))


Iwheel = cv2.imread(os.path.join(max_activations_dir, 'color_wheel.jpg'))
cv2.imshow('wheel', Iwheel)

caffe.set_mode_gpu()
caffe.set_device(2)
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

def viz_flow2(Is):
    h,w = Is[0].shape[:2]
    Iret = np.zeros((h, w*len(Is)), np.uint8)
    for index,Ix in enumerate(Is):
        Iret[:,index*w:index*w+w] = Ix
    return Iret


def viz_layer(net, layer, Iimg, Iflow, max_activations, filter_index):
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
    n_filters = len(max_activations[layer])
    while key != ord('q'):
        print 'filter:',filter_index

        diffs = net.blobs[layer].diff * 0
        diffs[0][filter_index] = net.blobs[layer].data[0,filter_index]
        net.deconv_from_layer(layer, diffs, zero_higher = True)
        deconv_data = net.blobs['data'].diff[0,...]*100
        c,h,w = deconv_data.shape[:]
        dmax = deconv_data.max()
        dmin = deconv_data.min()
        r = dmax - dmin
        print dmin, dmax, r
        if r > 0:
            deconv_data = 255*(deconv_data - dmin)/r
        else:
            deconv_data = 128 + deconv_data * 0
        deconv_data = deconv_data.astype(np.uint8)

        Ixs = list()
        Iys = list()
        Is = list()
        for i in range(c/2):
            Ix = deconv_data[2*i+0,...]
            Iy = deconv_data[2*i+1,...]
            Iys.append(Ix)
            Ixs.append(Iy)
            Ix = cv2.GaussianBlur(Ix, (5, 5), 0)
            Iy = cv2.GaussianBlur(Iy, (5, 5), 0)
            Is.append((Ix, Iy))
        Ideconv_x = viz_flow2(Ixs)
        Ideconv_y = viz_flow2(Iys)
        Ideconv = viz_flow(Is)

        Idraw = Ires.copy()
        size = Ires.shape[0]/n
        y = size*(filter_index/n)
        x = size*(filter_index%n)
        cv2.rectangle(Idraw, (x,y), (x+size, y+size), (0,0,255), 2)
        cv2.imshow('response', Idraw)
        Ia = cv2.imread(max_activations[layer][filter_index])
        cv2.imshow('max_activations', Ia)
        Ib = cv2.imread(max_images[layer][filter_index])
        cv2.imshow('raw_images', Ib)
        cv2.imshow('filter response', cv2.resize(res_data[filter_index,...], (224, 224), interpolation = cv2.INTER_NEAREST))

        w = Ia.shape[1]
        h = w*Ideconv_x.shape[0]/Ideconv_x.shape[1]
        Ideconv_x = cv2.resize(Ideconv_x, (w,h))
        cv2.imshow('Ideconv_x', Ideconv_x)
        Ideconv_y = cv2.resize(Ideconv_y, (w,h))
        cv2.imshow('Ideconv_y', Ideconv_y)
        Ideconv = cv2.resize(Ideconv, (w,h))
        cv2.imshow('Ideconv', Ideconv)

        if Iflow.shape[1] != Ia.shape[1]:
            w = Ia.shape[1]
            h = w*Iflow.shape[0]/Iflow.shape[1]
            Iflow = cv2.resize(Iflow, (w,h))
        cv2.imshow('flow', Iflow)
        if Iimg.shape[1] != Ia.shape[1]:
            w = Ia.shape[1]
            h = w*Iimg.shape[0]/Iimg.shape[1]
            Iimg = cv2.resize(Iimg, (w,h))
        cv2.imshow('image', Iimg)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            sys.exit(0)
        new_index = filter_index
        if key == ord('l'):
            print 'l'
            new_index = filter_index + 1
        if key == ord('j'):
            new_index = filter_index - 1
        if key == ord('i'):
            new_index = filter_index - n
        if key == ord('k'):
            new_index = filter_index + n
        if new_index >= 0 and new_index < n_filters:
            filter_index = new_index

    return filter_index

max_data = pickle.load(open('plus44.pkl'))
k = max_data['conv5']
filter_index = 0
while True:
    entries = max_data['conv5'][filter_index]
    for entry in entries:
        flow_folder = entry[1]
        img_folder = os.path.join(images_dir, os.path.basename(flow_folder))
        tokens = os.path.splitext(entry[2])[0].split('_')
        start = int(tokens[-1])
        crop_size = 224
        mean = 128.0
        input_data = np.zeros((1, 10*2, crop_size, crop_size), np.float64)
        Is = deque()
        Iimgs = deque()
        for frame in range(start, start+10):
            basename = '_'.join(tokens[:-1]) + '_{:06d}.jpg'.format(frame)
            Ix = cv2.imread(os.path.join(flow_folder, 'x', basename))
            Iy = cv2.imread(os.path.join(flow_folder, 'y', basename))    
            x = (Ix.shape[1] - crop_size)/2
            y = (Ix.shape[0] - crop_size)/2
            Ix = Ix[y:y+crop_size, x:x+crop_size, 0].astype(np.float64) - mean
            Iy = Iy[y:y+crop_size, y:y+crop_size, 0].astype(np.float64) - mean
            I = cv2.imread(os.path.join(img_folder, basename))
            Iimgs.append(I)
            Is.append((Ix, Iy))

        for i, (Ix, Iy) in enumerate(Is):
            input_data[0,i*2+0,...] = Ix
            input_data[0,i*2+1,...] = Iy

        net.blobs['data'].data[...] = input_data
        net.forward()
        Iimg = viz_images(Iimgs)
        Iflow = viz_flow(Is)
        filter_index = viz_layer(net, 'conv5', Iimg, Iflow, max_activations, filter_index)

sys.exit(0)


filter_index = 0
for (flow_folder,label) in flow_folders[518:]:
    print flow_folder
    x_folder = os.path.join(flow_folder, 'x')
    y_folder = os.path.join(flow_folder, 'y')
    x_images = sorted([os.path.join(x_folder, f) for f in os.listdir(x_folder)])
    y_images = sorted([os.path.join(y_folder, f) for f in os.listdir(y_folder)])

    num_stack_frames = 10
    crop_size = 224
    mean = 128.0
    input_data = np.zeros((1, num_stack_frames*2, crop_size, crop_size), np.float64)
    Is = deque()
    Iimgs = deque()
    for (x_img, y_img) in zip(x_images, y_images):
        Ix = cv2.imread(x_img)
        Iy = cv2.imread(y_img)
        x = (Ix.shape[1] - crop_size)/2
        y = (Ix.shape[0] - crop_size)/2
        Ix = Ix[y:y+crop_size, x:x+crop_size, 0].astype(np.float64) - mean
        Iy = Iy[y:y+crop_size, y:y+crop_size, 0].astype(np.float64) - mean
        I = cv2.imread(os.path.join(images_dir, os.path.basename(flow_folder), os.path.basename(x_img)))
        Iimgs.append(I)
        Is.append((Ix, Iy))

        if len(Is) == num_stack_frames:
            for i, (Ix, Iy) in enumerate(Is):
                input_data[0,i*2+0,...] = Ix
                input_data[0,i*2+1,...] = Iy

            net.blobs['data'].data[...] = input_data
            net.forward()
            score = net.blobs['score'].data.ravel()
            #print label, np.argmax(score)

            Iimg = viz_images(Iimgs)
            Iflow = viz_flow(Is)
            filter_index = viz_layer(net, 'conv5', Iimg, Iflow, max_activations, filter_index)

            Is.popleft()
            Iimgs.popleft()



