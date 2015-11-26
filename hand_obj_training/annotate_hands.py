'''Annotate manipulation point
'''

import os,sys
import cv2

def onMouseCallback(event, x, y, flags, param):
	global hand_index
	global hands
	if event == cv2.EVENT_LBUTTONUP:
		colors = [(0,0,255), (0,255,0)]

		hands[hand_index] = (x,y)

		Idraw = I.copy()
		for i, (x,y) in enumerate(hands):
			cv2.circle(Idraw, (x, y), 5, colors[i], -1)
		cv2.imshow('image', Idraw)

		hand_index += 1
		hand_index = hand_index % 2

img_dir = 'raw_data/img'
gt_dir = 'raw_data/mask'

images = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
cv2.namedWindow('image')
cv2.setMouseCallback('image', onMouseCallback)

hand_index = 0
hands = [(-1, -1), (-1, -1)]
fw = open('hands.txt', 'w')
for image in images:
	hand_index = 0
	hands = [(-1, -1), (-1, -1)]
	I = cv2.imread(image)
	Idraw = I.copy()
	cv2.imshow('image', Idraw)

	key = 0
	while key != 27 and key != 10:
		key = cv2.waitKey(0) & 0xFF
		if key == ord('c'):
			hand_index = 0
			hands = [(-1, -1), (-1, -1)]
			Idraw = I.copy()
			cv2.imshow('image', Idraw)
		if key == ord('n'):
			hand_index += 1
			if hand_index > 1:
				hand_index = 0

	if key == 27:
		break

	line = os.path.basename(image)
	for xy in hands:
		line = line + ' ' + str(xy[0]) + ' ' + str(xy[1])
	line += '\n'
	print line,
	fw.write(line)
	fw.flush()


fw.close()