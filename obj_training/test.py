caffe_root = '/home/minghuam/caffe-fcn'
import sys,os,shutil
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse

def predict_one_image(net, Iraw, Imean):
    I = cv2.resize(Iraw, (256, 256))
    Inorm = I - Imean
    net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
    out = net.forward()
    
    e_score = np.exp(out['score'])
    e_score_sum = e_score.sum(axis = 1).reshape((1, 1, 256, 256))
    e_score_sum = e_score_sum.transpose((1, 0, 2, 3)).repeat(repeats = 2, axis = 1)
    Iout =  e_score/e_score_sum
    Iout = (Iout[:,1,...]*255).astype(np.uint8).reshape((256, 256, 1))
    #Iout = np.repeat(Iout, repeats = 3, axis = 2)

    Iout = cv2.resize(Iout, (Iraw.shape[1], Iraw.shape[0]))

    return Iout

caffe.set_mode_gpu()
caffe.set_device(2)
net_proto_file =  'fcn32_obj_deploy.prototxt'
model_file = 'model/OBJ_iter_1000.caffemodel'
net = caffe.Net(net_proto_file, model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

K = 1
Imean = np.ones((K, 256, 256, 3), np.float32)
for c in range(K):
    for i in range(3):
        Imean[c, :,:,i] *= bgr_mean[i]

cap = cv2.VideoCapture('hand.avi')
while(cap.isOpened()):
    ret, Iraw = cap.read()
    if Iraw is None:
        break

    Iobj = predict_one_image(net, Iraw, Imean)

    unused, Ithresh = cv2.threshold(Iobj, 64, 255, 0)
    # cv2.imshow('Ithresh', Ithresh)

    contours, hierarchy = cv2.findContours(Ithresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_index = -1
    max_area = -1
    min_area = 100
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area < area:
            max_index = i
            max_area = area

    left = -1
    top = -1
    obj_size = 300
    if max_index != -1:
        M = cv2.moments(contours[max_index])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        top = cy - obj_size/2
        bottom = cy + obj_size/2 - 1
        if top < 0:
            top = 0
        if bottom > Iobj.shape[0] - 1:
            top = Iobj.shape[0] - 1 - obj_size

        left = cx - obj_size/2
        right = cx + obj_size/2 - 1
        if left < 0:
            left = 0
        if right > Iobj.shape[1] - 1:
            left= Iobj.shape[1] - 1 - obj_size

    # cv2.imshow('Ithresh', Ithresh)
    # cv2.imshow('Iprob', Iobj)

    Iobj = np.repeat(Iobj.reshape(Iobj.shape + (1,)), 3, axis = 2)
    Iobj[:,:,0] = 0
    Iobj[:,:,1] = 0

    if left != -1:
        cv2.rectangle(Iobj, (left, top), (left + obj_size, top + obj_size), (0, 255, 0), 2)
        y = int(Iraw.shape[0] * float(top)/Iobj.shape[0])
        x = int(Iraw.shape[1] * float(left)/Iobj.shape[1])
        size = int(Iraw.shape[0] * float(obj_size)/Iobj.shape[0])
        if x + size > Iraw.shape[1] - 1:
            x = Iraw.shape[1] - 1 - size
        if y + size > Iraw.shape[0] - 1:
            y = Iraw.shape[0] - 1 - size
        Icrop = Iraw[y:y+size, x:x+size, :]
        #Icrop = cv2.resize(Icrop, (256, 256))
        cv2.imshow('Iobj', Icrop)

    I =  cv2.addWeighted(Iraw, 0.75, Iobj, 0.5, 0)
    cv2.imshow('I', I)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()