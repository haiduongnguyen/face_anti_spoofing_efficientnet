# package: install numpyencoder
import numpy as np
import cv2
import os
import glob
import time
import json
from numpyencoder import NumpyEncoder
import sys
sys.path.insert(1, '/home/duong/project/pyimage_research/livenessDetection')
from config import helper_train_json_path, helper_test_json_path, detector, folder_save_json


print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


# load helper json file to read data
helper_train_json = json.load(open(helper_train_json_path))
helper_test_json = json.load(open(helper_test_json_path))


bb_label_test = {}
count = 0
for image_path in helper_test_json.keys():
    frame = cv2.imread(image_path)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # careful with crop image, Y first, X later
            # face = frame[startY:endY, startX:endX]
            count += 1
            if count%1000 == 0:
                print(count)
            label = helper_test_json[image_path][39]
            value = [startX, startY, endX, endY, label]
            bb_label_test.update({image_path: value})
print(f"complete detect {count} face to test")

with open(os.path.join(folder_save_json, 'test_label.json'), 'w') as outfile:
    json.dump(bb_label_test, outfile,cls=NumpyEncoder)
print('complete write test json file')

bb_label_train = {}
count = 0
for image_path in helper_train_json.keys():
    frame = cv2.imread(image_path)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    # ensure at least one face was found
    if len(detections) > 0:
        # we're making the assumption that each image has only ONE
        # face, so find the bounding box with the largest probability
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections)
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the face and extract the face ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # careful with crop image, Y first, X later
            # face = frame[startY:endY, startX:endX]
            count += 1
            if count%5000 == 0:
                print(count)
            label = helper_train_json[image_path][39]
            value = [startX, startY, endX, endY, label]
            bb_label_train.update({image_path: value})
print(f"complete detect {count} face to train")

with open(os.path.join(folder_save_json, 'train_label.json'), 'w') as outfile:
    json.dump(bb_label_train, outfile, cls=NumpyEncoder)
print('complete write train json file')

print('done')