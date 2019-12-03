# Dog-Breed and Face-Detection 
This is an image classification project for my deep learning nanodegree at Udacity.

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

 
