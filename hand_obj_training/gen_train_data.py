import os,sys,cv2,shutil

img_dir = 'raw_data/img'
msk_dir = 'raw_data/mask'
obj_dir = 'raw_obj'
hand_dir = 'raw_hands'

output_dir = 'data'

imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
msks = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir)])
objs = sorted([os.path.join(obj_dir, f) for f in os.listdir(obj_dir)])
hands = sorted([os.path.join(hand_dir, f) for f in os.listdir(hand_dir)])

if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, 'img'))
os.mkdir(os.path.join(output_dir, 'msk'))
os.mkdir(os.path.join(output_dir, 'hand_obj'))

with open('data.txt', 'w') as fw:
	for (img, msk, obj, hand) in zip(imgs, msks, objs, hands):
		print img, msk, obj
		basename = os.path.basename(img)
		h = 256
		w = 256
		Iimg = cv2.resize(cv2.imread(img), (h,w))
		Imsk = cv2.resize(cv2.imread(msk), (h,w))
		Iobj = cv2.resize(cv2.imread(obj), (h,w))
		Ihand = cv2.resize(cv2.imread(hand), (h,w))
		Ihand_obj = Ihand + Iobj

		img = os.path.abspath(os.path.join(output_dir, 'img', basename))
		msk = os.path.abspath(os.path.join(output_dir, 'msk', basename))
		hand_obj = os.path.abspath(os.path.join(output_dir, 'hand_obj', basename))
		cv2.imwrite(img, Iimg)
		cv2.imwrite(msk, Imsk)
		cv2.imwrite(hand_obj, Ihand_obj)

		fw.write(img + ' ' + msk + ' ' + hand_obj + '\n')