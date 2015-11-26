import os,sys
import cv2
import numpy as np
import scipy

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = scipy.mgrid[-size:size+1, -sizey:sizey+1]
    g = scipy.exp(-(x**2/float(size*8)+y**2/float(sizey*8)))
    print g.shape
    return g / g.max()

Ig = (gauss_kern(500)*255).astype(np.uint8)
cv2.imshow('Ig', Ig)
#cv2.waitKey(0)
print Ig.shape

images_dir = 'raw_data/img'
output_dir = 'raw_obj'

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

with open('objects.txt', 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		image = tokens[0]
		x1 = int(tokens[1])
		y1 = int(tokens[2])
		
		I = cv2.imread(os.path.join(images_dir, image))
		Ih = np.zeros(I.shape, np.uint8)

		if x1 != -1:
			cv2.circle(Ih, (x1, y1), 5, (255, 0, 0), -1)

		k_size = (Ig.shape[0] - 1)/2
		Ih_p = np.zeros((I.shape[0] + Ig.shape[0]*2, I.shape[1] + Ig.shape[0]*2, 3), np.uint8)
		
		if x1 != -1:
			x1_p = x1 + Ig.shape[1]
			y1_p = y1 + Ig.shape[0]
			Ih_p[y1_p-k_size:y1_p+k_size+1, x1_p-k_size:x1_p+k_size+1, 0] = Ig

		Ih_c = Ih_p[Ig.shape[0]:Ig.shape[0]+I.shape[0], Ig.shape[1]:Ig.shape[1]+I.shape[1], :]

		cv2.imwrite(os.path.join(output_dir, image), Ih_c)

		cv2.imshow('I', I)
		cv2.imshow('Ih', Ih)
		cv2.imshow('Ih_p', Ih_c)
		if cv2.waitKey(30) & 0xFF == 27:
			break