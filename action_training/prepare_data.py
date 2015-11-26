import sys,os,cv2

flow_dir = 'flow'
obj_dir = 'obj'

flow_folders = sorted([os.path.join(flow_dir, d) for d in os.listdir(flow_dir)])
obj_folders = sorted([os.path.join(obj_dir, d) for d in os.listdir(obj_dir)])

action_ids = dict()
with open('action_ids.txt', 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		action_ids[tokens[0]] = tokens[1]

fw_action = open('action_data.txt', 'w')
fw_motion = open('motion_data.txt', 'w')
fw_spatial = open('spatial_data.txt', 'w')

for (flow_folder, obj_folder) in zip(flow_folders, obj_folders):
	print flow_folder, obj_folder
	basename = os.path.basename(flow_folder)
	action = '_'.join(basename.split('_')[:-1])
	Id = action_ids[action]
	fw_motion.write(os.path.abspath(flow_folder) + ' ' + Id + '\n')
	fw_action.write(os.path.abspath(flow_folder) + ' ' + os.path.abspath(obj_folder) + ' ' + Id + ' ' + Id + ' ' + Id + '\n')
	for img in sorted(os.listdir(obj_folder)):
		fw_spatial.write(os.path.abspath(os.path.join(obj_folder, img)) + ' ' + Id + '\n')

fw_action.close()
fw_motion.close()
fw_spatial.close()