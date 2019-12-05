# Overview
This project uses convolutional neural networks (**CNNs**) in Pytorch to classify images of humans and dogs. 
Humans are recognized using a face detector and dogs using a network pretrained on the [ImageNet dataset](http://www.image-net.org/).
Finally the dogs are classified based on their breed and an output is return to the user.
Humans can be classified as well, in case you want to know to which dog breed you look more similar!

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

## Step1: Datasets
The two datasets have the following characteristics:
* The [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip) includes 13233 images of various size. Most pictures are faces of personalities available online.
* The [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)includes 8351 images of various size of dogs organised in 133 breeds. Some of the images include humans (their faces or body parts).

## Step2: Face Detection
Human faces are detected using a [Haar feature-detection cascade classifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html) available from the [OpenCV](https://opencv.org) library.
Faces were sought in both datasets to assess the quality of the detection as the classifier is known to have issues recognising faces in certain conditions, e.g. in non-frontal face pictures.
 
## Step3: Detect Dogs
A [VGG-16 network](https://arxiv.org/abs/1409.1556) pretrained on the ImageNet dataset is used for the dog detection task.
The ImageNet dataset __includes__ dog breeds between its [1000 classes](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), but they are only 188 (class nr:151 to 268).
As in Step2, the detector is applied to both the human and dog datasets to assess its performance.
The same step was performed comparing these network architectures, [available](https://pytorch.org/docs/stable/torchvision/models.html) from torchvision:
* [Inception-v3](https://arxiv.org/abs/1512.00567) 
* [ResNet](https://arxiv.org/abs/1512.03385) 
* [DenseNet](https://arxiv.org/abs/1608.06993)

## Step4 Design and implement a CNN to classify dog breeds
A CNN was designed from scratch, trained and tested for recognizing dog breeds.
These are the main points to highlight:
1. **Image preprocessing**:  this CNN resizes the images to 224x224 pixels (RGB color channels) and augments the the training set using random resize crop, color jittering, random horizontal flip and random rotations. 
Finally it applies the standard normalization for ImageNet inputs using `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.
2. **Architecture**: The CNN is designed to be:
	* as deep as possible, within the available GPU memory limits (three convolutional layers with ReLU activation, each followed by MaxPooling). Maximum number of channels is 128.
	* halve the HxW dimension of the feature maps at each layer.
	* finish with two fully connected layers with a dropout layer to prevent overfitting 
	* the final layer outputs scores to a softmax function providing dog-breed class probabilies.
2. **Training**: the network was trained until the validation loss did not decrease (around 50 epochs) using:
	* using a **cross-entropy** loss
	* using an **Adam optimizer** with an adaptive learning rate starting at 0.001 and decreasing by a factor of 10 every time the loss was not decreasing for six epochs
	* mini-batches of 128 images	

## Step5 Create a CNN to classify dog breeds using transfer learning
The same VGG-16 network used at Step3 for the dog detection was modified and used to identify their breeds.
* The architecture was modified just at the final two fully connected layers to match the number of classes (133) and decrease the number of parameters.
* The training set augmentation and training set up was the same as for the CNN trained from scratch.
* The network was trained for around 10 epochs.

## Step6-7 Write and test an algorithm to read an image and apply the previous points
An algorithm was developed that uses all the functions produced at the steps above and does the following:
1. detects if dog, human, dog and human or neither of them are present in an image given as input.
2. plots the image with a greeting message appropriate to the identified content
3. plots the top-5 classes and their probabilities.
Note that the CNNs used in this implementation is that from Step5 because the achieved accuracy was much larger (>70% versus 22%).

# Limitations
* Haare classifier does not recognise faces in all images.
* The entire workflow is a bit "cranky" to run at the moment.

# Future work
OpenCV offers some deep-learning algoorithms for face detection in their [dnn module](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector).
Those models can be implemented in Cafee using:
1. a Caffee model file `deploy.prototxt` 
2. network trained weights `weights.meta4`. 
Those files are in the `opencv_files` directory and there is some initial test [at this link](https://github.com/modenaxe/dog-breed-classifier/tree/master/opencv_dnn_face_detector) .

# Other Resources used in this project
* [Git Large File Storage](https://git-lfs.github.com/)
 
