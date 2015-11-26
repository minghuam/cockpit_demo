import os,sys,cv2,shutil

img_dir = 'raw_data/img'
msk_dir = 'raw_data/mask'
obj_dir = 'raw_obj'

output_dir = 'data'

imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
msks = sorted([os.path.join(msk_dir, f) for f in os.listdir(msk_dir)])
objs = sorted([os.path.join(obj_dir, f) for f in os.listdir(obj_dir)])

if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, 'img'))
os.mkdir(os.path.join(output_dir, 'msk'))
os.mkdir(os.path.join(output_dir, 'obj'))

with open('data.txt', 'w') as fw:
	for (img, msk, obj) in zip(imgs, msks, objs):
		print img, msk, obj
		basename = os.path.basename(img)
		h = 256
		w = 256
		Iimg = cv2.resize(cv2.imread(img), (h,w))
		Imsk = cv2.resize(cv2.imread(msk), (h,w))
		Iobj = cv2.resize(cv2.imread(obj), (h,w))

		img = os.path.abspath(os.path.join(output_dir, 'img', basename))
		msk = os.path.abspath(os.path.join(output_dir, 'msk', basename))
		obj = os.path.abspath(os.path.join(output_dir, 'obj', basename))
		cv2.imwrite(img, Iimg)
		cv2.imwrite(msk, Imsk)
		cv2.imwrite(obj, Iobj)

		fw.write(img + ' ' + msk + ' ' + obj + '\n')