## 1. Hand segmentation
The main algorithm is from the FCN paper: [Fully convolutional networks for semantic segmentation](http://arxiv.org/abs/1411.4038). 

### Data preparation
1. Change current directory to *hand_training*. See raw hand annotation data is in *raw_data*.
2. Run *prepare_data.py* to resize images and hand masks to 256x256. This script also generates *data.txt*. Each line of this file is a pair of image and mask sperated by space. Ignore the two ending numbers.

### Training
1. Pull my modified caffe from here: [caffe-fcn](https://github.com/minghuam/caffe-fcn).
2. Modify the caffe root path in *fcn32_solve.py*.
3. Run *train.sh* to train the network. Models will be saved in *model* folder. See *fcn32_solve.py* for training details.

### Testing
See *test.py*. This script reads frames from video and predicts a hand probability map.

## 2. Object localization
With the hand segmentation model, we fine-tune it to localize the object of interest. Code and data is in *obj_training*.

### Data preparation
1. Copy the same hand annotation data(image+mask)  into *obj_training/raw_data* (optional, can still use the same data source for hand segmentation training). Run *annotate_obj.py* and annotate object centers. This script will create a text file *objects.txt* with object location coordinates.
2. Run *cal_heatmap.py* to generate object heatmaps. Heatmaps will be save in *raw_obj*.
3. Run *gen_train_data.py* to resize images, mask and heatmaps. Final data will be saved in *data* folder. It also generates training source file *data.txt*. 

### Training
1. Modify the caffe root path in *fcn32_solve.py*.
2. Run *train.sh* to train the network. Models will be saved in *model* folder. See *fcn32_solve.py* for training details.

### Testing
See *test.py*. This script reads frames from video and predicts an object probability map.


## 3. Hand and objects localization
It is also possible to localize both object of interest and hand centers simutaneously. Code and data is in *hand_obj_training*.
The idea is the same as object localization except for number of channels in groundtruth data and network loss. 

For object localization only, object heatmaps are placed in the blue channel(1st channel in opencv). For this network, two additional channels are used: green channel for left hand  and red channel for right hand. See *data/hand_obj* for examples.

The other difference is network loss. For hand segmentation and object localization, the network output is a 2-channel probability map(background and hand/object). For this network, the output has 5 channels which are splitted into 2+3 channels. They are for (1) background for object (2) object (3) background for hands (4) left hand (5) right hand respectively. See *fcn32_obj_train.prototxt* for details.

## 4. Two-stream action recognition
Currently, this relies on a different version of caffe. Pull it here: [caffe-dev](https://github.com/minghuam/caffe-dev). Code and data is in *action_training*.

### Object network
Before training the spatial network, you need to crop the objects out from training data. See *crop_obj.py*. This scripts loads the trained object localization model and saves images in *obj*. All network files and training scripts are in *spatial*. 

### Motion network
To calculate optical flow, you can use scripts in *tools*. Save optical flow images in *flow* and then run training scripts in *motion* to train the network.

### Joint training
See *action_training/solve.py*.