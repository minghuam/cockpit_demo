import os, sys
import cv2
import argparse
from multiprocessing import Process


def vid_to_images(video_file, output_dir, skip_frames = 0):
    
    print 'Processing video:', video_file
    
    cap = cv2.VideoCapture(video_file)
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    image_dir = os.path.join(output_dir, video_name)
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)    

    frame = 0
    while(cap.isOpened()):
        ret, I = cap.read()    
        if not ret:
            break

        #I = cv2.resize(I, (320, 240))
        #I = cv2.resize(I, (360, 203))
        #height, width = I.shape[0:2]
        #x = (width - height)/2
        #I = I[:, x:x+height, :]
        #I = cv2.resize(I, (256, 256))

        #cv2.imshow('I', I)
        #if cv2.waitKey(0) & 0xFF == 27:
        #    break

        if frame % (skip_frames + 1) == 0:
            image_filename = "{}_{:06d}.jpg".format(video_name, frame + 1)
            cv2.imwrite(os.path.join(image_dir, image_filename), I)
        frame += 1
    
    cap.release()   
    print 'Done with', video_file, ', total frames:', frame + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video_file', help = 'Video file path')
    parser.add_argument('-d', '--video_dir', help = 'Video files directory')
    parser.add_argument('-s', '--skip_frames', type = int, help = 'Number of frames to skip')
    parser.add_argument('output_dir', help = 'Output directory')
    args = parser.parse_args()
    
    if args.video_file is None and args.video_dir is None:
        parser.print_help()
        sys.exit(0)

    skip_frames = 0
    if args.skip_frames:
        skip_frames = args.skip_frames

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.video_file:
        vid_to_images(args.video_file, args.output_dir, skip_frames)

    if args.video_dir:
        video_files = [os.path.join(args.video_dir, v) for v in os.listdir(args.video_dir) if v.endswith('.avi') or v.endswith('.mp4') or v.endswith('.mpg')]
        num_videos = len(video_files)
        print 'Total number of videos:', num_videos
    
        num_procs = 4
        index = 0
        while index < num_videos:
            procs = []
            for i in range(index, index + num_procs):
                if i < num_videos:
                    p = Process(target = vid_to_images, args = (video_files[i], args.output_dir, skip_frames,))
                    procs.append(p)
                    p.start()
            for p in procs:
                p.join()
            index += num_procs
                










