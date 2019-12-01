import numpy as np
import cv2

# used: https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/

# loading the DL model
prototxt_file = './opencv_files/deploy.prototxt'
caffe_model = './opencv_files/res10_300x300_ssd_iter_140000_fp16.caffemodel'

# in alternative there is openface
# https://jeanvitor.com/tensorflow-object-detecion-opencv/

net = cv2.dnn.readNetFromCaffe(prototxt_file, caffe_model)

# loading the image
img_path = "./test_images/pitt.jpeg"
img = cv2.imread(img_path)
(h, w) = img.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 3000), (104.0, 177.0, 123.0))

# 
net.setInput(blob)
detections = net.forward()


for i in range (0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        text = "{:.2f}%".format(confidence*100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(img, (startX, startY), (endX, endY),(0, 0, 255), 2)
        cv2.putText(img, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
# show the output image
cv2.imshow("Output", img)
cv2.waitKey(0)
#print(detections)