import os,sys,cv2
import numpy as np

def createColorWheel(r):
	d = 2*r + 1
	xx,yy = np.meshgrid(np.arange(d), np.arange(d))

	angles = np.arctan2(r-yy, xx-r)
	dists = np.sqrt(np.power(xx - r, 2) + np.power(yy - r, 2))

	Hmax = 179
	H = ((angles + np.pi)/(2*np.pi)*Hmax).astype(np.uint8)
	S = (np.clip(dists/r, 0, 1)*255).astype(np.uint8)

	HSV = np.ones((2*r+1, 2*r+1, 3), np.uint8)*255
	HSV[...,0] = H
	HSV[...,1] = S

	I = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)

	return I

Iwheel = createColorWheel(200)
cv2.imshow('Iwheel', Iwheel)

def ls_images(d):
	return sorted([os.path.join(d, f) for f in os.listdir(d)])

def viz_flow(Ix, Iy):
	Ix = Ix - 128.0
	Iy = Iy - 128.0

	angles = np.arctan2(Iy, Ix)
	dists = np.sqrt(np.power(Ix, 2), np.power(Iy, 2))

	Hmax = 179
	H = ((angles + np.pi)/(2*np.pi)*Hmax).astype(np.uint8)
	Dmax = np.sqrt(127*127 + 127*127)
	dists *= 5
	S = (np.clip(dists/Dmax, 0, 1)*255).astype(np.uint8)
	HSV = np.ones((Ix.shape[0], Ix.shape[1], 3), np.uint8)*255
	HSV[...,0] = H
	HSV[...,1] = S

	I = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR)
	return I	

flow_dir = '/home/minghuam/data/GTEA_gaze_plus/RAW/flow44/Ahmad_take_milk_container_0390/'
x_folder = os.path.join(flow_dir, 'x')
y_folder = os.path.join(flow_dir, 'y')
for (x_img, y_img) in zip(ls_images(x_folder), ls_images(y_folder)):
	print x_img, y_img
	Ix = cv2.imread(x_img, cv2.CV_LOAD_IMAGE_GRAYSCALE)
	Iy = cv2.imread(y_img, cv2.CV_LOAD_IMAGE_GRAYSCALE)

	I = viz_flow(Ix, Iy)
	cv2.imshow('I', I)

	key = 0xFF & cv2.waitKey(0)
	if key == 27:
		sys.exit(0)
	if key == ord('s'):
		cv2.imwrite('flow.jpg', I)