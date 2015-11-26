import sys,os,cv2,shutil

raw_img_dir = 'raw_data/img'
raw_msk_dir = 'raw_data/mask'

output_dir = 'data'

if os.path.exists(output_dir):
	shutil.rmtree(output_dir)
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, 'img'))
os.mkdir(os.path.join(output_dir, 'msk'))

imgs = [os.path.join(raw_img_dir, f) for f in os.listdir(raw_img_dir)]
msks = [os.path.join(raw_msk_dir, f) for f in os.listdir(raw_msk_dir)]

if len(imgs) !=  len(msks):
	print 'image and mask are not equal!'
	sys.exit(0)

with open('data.txt', 'w') as fw:
	for (img, msk) in zip(imgs, msks):
		print img, msk
		if os.path.basename(img) != os.path.basename(msk):
			print 'image and mask mismatch!'
			sys.exit(0)

		Iimg = cv2.imread(img)
		Imsk = cv2.imread(msk)

		width = 256
		height = 256
		Iimg = cv2.resize(Iimg, (width, height))
		Imsk = cv2.resize(Imsk, (width, height))
		basename = os.path.basename(img)
		cv2.imwrite(os.path.join(output_dir, 'img', basename), Iimg)
		cv2.imwrite(os.path.join(output_dir, 'msk', basename), Imsk)

		fw.write(os.path.abspath(os.path.join(output_dir, 'img', basename)) + ' ' + os.path.abspath(os.path.join(output_dir, 'msk', basename)) + ' 0.0 0.0\n')
		
		cv2.imshow('Image', Iimg)
		cv2.imshow('Imsk', Imsk)
		if cv2.waitKey(30) & 0xFF == 27:
			sys.exit(0)