'''Annotate object point
'''

import os,sys
import cv2

def onMouseCallback(event, x, y, flags, param):
	global object_pos
	if event == cv2.EVENT_LBUTTONUP:
		colors = [(0,0,255), (0,255,0)]
		object_pos = (x, y)
		Idraw = I.copy()
		cv2.circle(Idraw, (x, y), 5, (0, 0, 255), -1)
		cv2.imshow('image', Idraw)

img_dir = 'raw_data/img'
gt_dir = 'raw_data/mask'

images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouseCallback)

object_pos = (-1, -1)
fw = open('objects.txt', 'w')
for image in images:
	object_pos = (-1, -1)
	I = cv2.imread(image)
	Idraw = I.copy()
	cv2.imshow('image', Idraw)

	key = 0
	while key != 27 and key != 10:
		key = cv2.waitKey(30) & 0xFF
		if key == ord('c'):
			object_pos = (-1, -1)
			Idraw = I.copy()
			cv2.imshow('image', Idraw)

	if key == 27:
		break

	line = os.path.basename(image) + ' ' + str(object_pos[0]) + ' ' + str(object_pos[1]) + '\n'
	print line,
	fw.write(line)
	fw.flush()


fw.close()