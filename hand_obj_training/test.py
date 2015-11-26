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
    
    e_score = np.exp(out['object_score'])
    e_score_sum = e_score.sum(axis = 1).reshape((1, 1, 256, 256))
    e_score_sum = e_score_sum.transpose((1, 0, 2, 3)).repeat(repeats = 2, axis = 1)
    Iout =  e_score/e_score_sum
    Iobj = (Iout[:,1,...]*255).astype(np.uint8).reshape((256, 256, 1))    
    Iobj = cv2.resize(Iobj, (Iraw.shape[1], Iraw.shape[0]))

    e_score = np.exp(out['hand_score'])
    e_score_sum = e_score.sum(axis = 1).reshape((1, 1, 256, 256))
    e_score_sum = e_score_sum.transpose((1, 0, 2, 3)).repeat(repeats = 3, axis = 1)
    Iout =  e_score/e_score_sum
    Ileft = (Iout[:,1,...]*255).astype(np.uint8).reshape((256, 256, 1))
    Iright = (Iout[:,2,...]*255).astype(np.uint8).reshape((256, 256, 1))    
    Ileft = cv2.resize(Ileft, (Iraw.shape[1], Iraw.shape[0]))
    Iright = cv2.resize(Iright, (Iraw.shape[1], Iraw.shape[0]))

    return Iobj, Ileft, Iright


def get_bbox(Iprob, threshold_val, obj_size):

    unused, Ithresh = cv2.threshold(Iprob, threshold_val, 255, 0)
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
    #obj_size = 300
    if max_index != -1:
        M = cv2.moments(contours[max_index])
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        top = cy - obj_size/2
        bottom = cy + obj_size/2 - 1
        if top < 0:
            top = 0
        if bottom > Iprob.shape[0] - 1:
            top = Iprob.shape[0] - 1 - obj_size

        left = cx - obj_size/2
        right = cx + obj_size/2 - 1
        if left < 0:
            left = 0
        if right > Iprob.shape[1] - 1:
            left= Iprob.shape[1] - 1 - obj_size

    size = -1
    if left != -1:
        cv2.rectangle(Iobj, (left, top), (left + obj_size, top + obj_size), (0, 255, 0), 2)
        y = int(Iraw.shape[0] * float(top)/Iobj.shape[0])
        x = int(Iraw.shape[1] * float(left)/Iobj.shape[1])
        size = int(Iraw.shape[0] * float(obj_size)/Iobj.shape[0])
        if x + size > Iraw.shape[1] - 1:
            x = Iraw.shape[1] - 1 - size
        if y + size > Iraw.shape[0] - 1:
            y = Iraw.shape[0] - 1 - size
        left = x
        top = y
    return (left, top, left + size, top + size)


caffe.set_mode_gpu()
caffe.set_device(0)
net_proto_file =  'fcn32_obj_deploy.prototxt'
model_file = 'model/OBJ_iter_4000.caffemodel'
net = caffe.Net(net_proto_file, model_file, caffe.TEST)
bgr_mean = [104.00699, 116.66877, 122.67892]

K = 1
Imean = np.ones((K, 256, 256, 3), np.float32)
for c in range(K):
    for i in range(3):
        Imean[c, :,:,i] *= bgr_mean[i]

cap = cv2.VideoCapture('hand.avi')
fourcc = cv2.cv.CV_FOURCC(*'MP4V')
video_writer = None
while(cap.isOpened()):
    ret, Iraw = cap.read()
    if Iraw is None:
        break

    if video_writer is None:
        video_writer = cv2.VideoWriter('output.avi', fourcc, 15, (Iraw.shape[1], Iraw.shape[0]))

    Iobj, Ileft, Iright = predict_one_image(net, Iraw, Imean)

    I_hand_obj = np.zeros(Iraw.shape, np.uint8);
    I_hand_obj[:,:,0] = Iobj
    I_hand_obj[:,:,1] = Ileft
    I_hand_obj[:,:,2] = Iright

    obj_bbox = get_bbox(Iobj, 128, 300)
    left_bbox = get_bbox(Ileft, 128, 200)
    right_bbox = get_bbox(Iright, 128, 200)

    #I =  cv2.addWeighted(Iraw, 0.75, Iobj, 0.5, 0)
    Iraw =  cv2.addWeighted(Iraw, 0.75, I_hand_obj, 0.5, 0)
    #cv2.imshow('I', I)

    if obj_bbox[0] != -1:
        cv2.rectangle(Iraw, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), (255, 0, 0), 2)

    if left_bbox[0] != -1:
        cv2.rectangle(Iraw, (left_bbox[0], left_bbox[1]), (left_bbox[2], left_bbox[3]), (0, 255, 0), 2)

    if right_bbox[0] != -1:
        cv2.rectangle(Iraw, (right_bbox[0], right_bbox[1]), (right_bbox[2], right_bbox[3]), (0, 0, 255), 2)
    cv2.imshow('I', Iraw)

    video_writer.write(Iraw)

    if cv2.waitKey(10) & 0xFF == 27:
        break

video_writer.release()

cap.release()
cv2.destroyAllWindows()
