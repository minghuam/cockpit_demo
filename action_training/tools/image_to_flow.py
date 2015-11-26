"""Calculate and save optical flow map files.
"""

import os, sys
import subprocess
import argparse
from multiprocessing import Process

def ls_images(folder_path, extension = 'png'):
    return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.' + extension)])

def ls_directores(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f[0] != '.']
    return sorted([f for f in files if os.path.isdir(f)])

def mkdir_new(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def proc_flow(cpp_tool, imgs_dir, output_dir):
    flow_dir = os.path.join(output_dir, os.path.basename(imgs_dir))
    flow_x_dir = os.path.join(flow_dir, 'x')
    flow_y_dir = os.path.join(flow_dir, 'y')
    if not os.path.exists(flow_dir):
        os.mkdir(flow_dir)
    if not os.path.exists(flow_x_dir):
        os.mkdir(flow_x_dir)
    if not os.path.exists(flow_y_dir):
        os.mkdir(flow_y_dir)

    args = (cpp_tool, '-i='+imgs_dir, '-x='+flow_x_dir, '-y='+flow_y_dir, '-b=2.5', '-t=1', '-d=1', '-s=0')
    subprocess.call(args)

    #images = ls_images(imgs_dir, extension = 'jpg')
    #num_images = len(images)
    #for i in xrange(num_images - 1):
    #    print "{}: {}/{}".format(images[i], i+1, num_images - 1)
    #    p = images[i]
    #    n = images[i+1]
    #    flow_x = os.path.join(flow_x_dir, os.path.basename(n))
    #    flow_y = os.path.join(flow_y_dir, os.path.basename(n))
    #    args = (cpp_tool, '-p='+p, '-n='+n, '-x='+flow_x, '-y='+flow_y, '-b=20', '-t=1', '-d=0', '-s=0')
    #    subprocess.call(args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('cpp_tool', help = 'cpp tool to calc flow')
    parser.add_argument('images_dir', help = 'images directory')
    parser.add_argument('flow_dir', help = 'output flow directory')
    args = parser.parse_args()

    if not os.path.exists(args.flow_dir):
        os.mkdir(args.flow_dir) 

    # for d in ls_directores(args.images_dir):
    #   proc_flow(args.cpp_tool, d, args.flow_dir)

    sub_images_dirs = ls_directores(args.images_dir)
    num_image_folders = len(sub_images_dirs)
    num_procs = 8
    index = 0
    while index < num_image_folders:
        procs = []
        for i in range(index, index + num_procs):
            if i < num_image_folders:
                print sub_images_dirs[i], '{}/{}'.format(i+1, num_image_folders)
                p = Process(target = proc_flow, args = (args.cpp_tool, sub_images_dirs[i], args.flow_dir))
                procs.append(p)
                p.start()
        for p in procs:
            p.join()
        index += num_procs
