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
    Iout = np.repeat(Iout, repeats = 3, axis = 2)

    Iout = cv2.resize(Iout, (Iraw.shape[1], Iraw.shape[0]))

    return Iout

caffe.set_mode_gpu()
caffe.set_device(0)
net_proto_file =  'fcn32_deploy.prototxt'
model_file = 'model/HAND_iter_2500.caffemodel'
net = caffe.Net(net_proto_file, model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

K = 1
Imean = np.ones((K, 256, 256, 3), np.float32)
for c in range(K):
    for i in range(3):
        Imean[c, :,:,i] *= bgr_mean[i]

cap = cv2.VideoCapture('weichiu.avi')
while(cap.isOpened()):
    ret, frame = cap.read()
    if frame is None:
        break

    Iout = predict_one_image(net, frame, Imean)

    Iout[...,0] = 0
    Iout[...,2] = 0

    I =  cv2.addWeighted(frame, 0.5, Iout, 0.5, 0)
    cv2.imshow('I', I)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
