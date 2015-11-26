import os,sys,cv2
import numpy as np
import pickle

output_dir = 'gtea_hand'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

max_activations = pickle.load(open('gtea_hand.pkl'))
for layer in max_activations:
    if layer.startswith('fc'):
        continue
    if layer != 'conv5':
        continue
    layer_dir = os.path.join(output_dir, layer)
    if not os.path.exists(layer_dir):
        os.mkdir(layer_dir)
    for filter_index in range(len(max_activations[layer])):
        entries = max_activations[layer][filter_index]
        n_entries = len(entries)
        n = int(np.ceil(np.sqrt(n_entries)))
        Itop = None
        for index in range(n_entries):
            entry = entries[-(index+1)]
            print entry
            I = cv2.imread(entry[1])
            x = int(entry[2])
            y = int(entry[3])
            crop_size = 224
            I = I[x:x+crop_size, y:y+crop_size, :]
            if Itop is None:
                h,w,c = I.shape[:]
                Itop = np.zeros((n*n,h,w,3), np.uint8)
            Itop[index,...] = I
        Itop = Itop.reshape((n, n*h, w, 3))
        Itop = Itop.transpose((1, 0, 2, 3))
        Itop = Itop.reshape((n*h, n*w, 3))
        cv2.imshow('I', Itop)
        cv2.waitKey(30)
        savename = '{}_{:04d}.jpg'.format(layer, filter_index)
        cv2.imwrite(os.path.join(layer_dir, savename), Itop)
