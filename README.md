# Dog-Breed and Face-Detection 

# Overview
In this project, I used convolutional neural networks to detect if an image contains a human face or a dog and, in the latter case, to identify the breed of that dog.
This is an image classification project for my deep learning nanodegree at Udacity.

# STEPS
## 1 Detect Human Faces

1. Using a Haar feature-detection cascade classifier
2. Using a deep learning based face detector from OpenCV
 
## 2 Detect Dogs (not breed)
1. Build a dog-detector using VGG-16 network pre-trained on Image-Net
2. Build a dog-detector using other architectures (Inception, ResNet and DenseNet)

## 3 Implement a dog-breed classifier
1. implementing a network from scratch
2. using a pretrained network and using transfer learning



## Face-detection
Face detection is performed in two ways:
1. using a Haare classifier, implemented by default in OpenCV
2. using a Caffe deep-learning network, also available from OpenCV

# Alternative face detection algorithm
OpenCV offers some deep-learning based algoorithm for face detection. There is a link on their main github [page](https://github.com/opencv/opencv/tree/ea667d82b30a19b10a6c00edf8acc6e9dd85c429/samples/dnn).

## Model
Model uses `dnn` model available from [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) and defined by:
1. a Caffee model file `deploy.prototxt` 
2. network trained weights `weights.meta4`. 
Those files are in the `opencv_files` directory.

# Other Sources
# https://www.pyimagesearch.com/start-here/
# https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/
# https://www.learnopencv.com/pytorch-for-beginners-semantic-segmentation-using-torchvision/

 
