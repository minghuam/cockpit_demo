import os,sys,cv2

img_dir = 'raw_data/img'
obj_dir = 'raw_obj'

imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
objs = sorted([os.path.join(obj_dir, f) for f in os.listdir(obj_dir)])

for (img,obj) in zip(imgs, objs):
	print img, obj
	I = cv2.imread(img)
	O = cv2.imread(obj)
	I = cv2.addWeighted(I, 0.75, O, 0.5, 0)

	cv2.imshow('I', I)
	if cv2.waitKey(0) & 0xFF == 27:
		break