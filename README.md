# Overview
This project uses convolutional neural networks (**CNNs**) in Pytorch to classify images of humans and dogs. 
Humans are recognized using a face detector and dogs using a network pretrained on the [ImageNet dataset](http://www.image-net.org/).
Finally the dogs are classified based on their breed and an output is return to the user.


# Project submission
This is a project for my deep-learning nanodegree at Udacity.
The submission files are [here](https://github.com/modenaxe/dog-breed-classifier/tree/master/submission).


# The Project
This project has multiple steps:
1. Load the dataset of human faces and dog breeds
2. Detect human faces in images
3. Detect dogs in images
4. Design and implement a CNN to classify dog breeds in [PyTorch](https://pytorch.org)
5. Create a CNN to classify dog breeds using transfer learning from an [existing model](https://pytorch.org/docs/stable/torchvision/models.html) 
6. Write an algorithm to read an image and apply the previous points
7. Test the algorithm on new images

## 1 Datasets
The two datasets have the following characteristics:
* The [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) includes 13233 images of various size. Most pictures are faces of personalities taken from the internet.
* The [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)includes 8351 images of various size of dogs organised in 133 breeds. Some of the images include humans (their faces or body parts).

## 2 Face Detection
* Human faces are detected using a [Haar feature-detection cascade classifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) available from the [OpenCV](https://opencv.org) library.
* Faces were sought in both datasets to assess the quality of the detection. The classifier is known to have issues recognising faces in certain conditions, especially in non-frontal pictures.
 
## 3 Detect Dogs
A [VGG-16 network](https://arxiv.org/abs/1409.1556) pretrained on the ImageNet dataset is used for this detection task.
The ImageNet dataset __includes__ dog breeds between its [1000 classes](https://gist.github.com/modenaxe/b00024740ed273bd30d700d2841aeaf5), but they are only 188 (class 151 to 268).
Dogs are detected in the images from both datasets using a
 

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

## Future work
2. Using a deep learning based face detector from OpenCV
Model uses `dnn` model available from [here](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector) and defined by:
1. a Caffee model file `deploy.prototxt` 
2. network trained weights `weights.meta4`. 
Those files are in the `opencv_files` directory.

# Other Resources used in this project
* [Git Large File Storage](https://git-lfs.github.com/)
* [pyimagesearch blog](https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/)
 
