import os,sys,cv2,shutil

input_dir = 'labeled_data'
raw_output_dir = 'raw_data'
resize_output_dir = 'resize_data'

if os.path.exists(raw_output_dir):
	shutil.rmtree(raw_output_dir)
os.mkdir(raw_output_dir)
if os.path.exists(resize_output_dir):
	shutil.rmtree(resize_output_dir)
os.mkdir(resize_output_dir)

subject_folders = sorted([os.path.join(input_dir, d) for d in os.listdir(input_dir)])

raw_folders = []
for subject_folder in subject_folders:
	folders = sorted([os.path.join(subject_folder, d) for d in os.listdir(subject_folder)])
	raw_folders = raw_folders + folders

print raw_folders
print len(raw_folders)

action_folders = dict()
folder_index = 0
frame_index = 0
for i, raw_folder in enumerate(raw_folders):
	print i+1, len(raw_folders), raw_folder
	action = '_'.join(os.path.basename(raw_folder).split('-')[:-1])
	resize_output_folder = os.path.join(resize_output_dir, '{}_{:06d}'.format(action, folder_index))
	raw_output_folder = os.path.join(raw_output_dir, '{}_{:06d}'.format(action, folder_index))
	os.mkdir(resize_output_folder)
	os.mkdir(raw_output_folder)
	if action not in action_folders:
		action_folders[action] = list()
	action_folders[action].append(os.path.abspath(resize_output_folder))

	imgs = sorted([os.path.join(raw_folder, f) for f in os.listdir(raw_folder)])
	for img in imgs:
		I = cv2.imread(img)
		cv2.imwrite(os.path.join(raw_output_folder, '{}_{:06d}.jpg'.format(action, frame_index)), I)
		I = cv2.resize(I, (256, 256))
		cv2.imwrite(os.path.join(resize_output_folder, '{}_{:06d}.jpg'.format(action, frame_index)), I)
		frame_index += 1
		cv2.imshow('I', I)
		if cv2.waitKey(10) & 0xFF == 27:
			sys.exit(0)

	folder_index += 1

action_ids = dict()
with open('action_ids.txt', 'w') as fw:
	index = 0
	actions = sorted(action_folders)
	for action in actions:
		fw.write(action + ' ' + str(index) + '\n')
		action_ids[action] = index
		index += 1