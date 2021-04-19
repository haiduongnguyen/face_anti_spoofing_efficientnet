import os
import json
import sys
import cv2
import numpy as np
from config import helper_train_json_path, helper_test_json_path, detector, crop_folder
from tqdm import tqdm

face_threshold = 0.7

print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def detect_face_and_crop(dictionary_from_json, mode , folder_output):
    saved = 0

    for key, value in tqdm(dictionary_from_json.items()):
        try:
            frame = cv2.imread(key)

            (h, w) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                (300, 300), (104.0, 177.0, 123.0))
            # pass the blob through the network and obtain the detections and
            # predictions
            net.setInput(blob)
            detections = net.forward()
            if len(detections) > 0:
                # we're making the assumption that each image has only ONE
                # face, so find the bounding box with the largest probability
                for i in range(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                # i = np.argmax(detections[0, 0, :, 2])
                # confidence = detections[0, 0, i, 2]

                if confidence > face_threshold:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face and extract the face ROI
                
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
        
                    if startX <= w and endX <= w and startY <= h and endY <= h:
                        # filter out weak detections
                        face = frame[startY:endY, startX:endX]
                        if 0 not in face.shape:
                            face = cv2.resize(face, (224,224))

                        if value[43] == 0:
                            p = os.path.join(folder_output, mode,'live',str(i) + partitions[-1])
                            if not cv2.imwrite(p, face):
                                raise Exception("Could not write image")
                            saved += 1
                        if value[43] == 1:
                            p = os.path.join(folder_output, mode,'spoof',str(i) + partitions[-1])
                            if not cv2.imwrite(p, face):
                                raise Exception("Could not write image")
                            saved += 1

        except:
            with open('broke_image.txt', 'a') as f:
                f.write('\n')
                f.write(key)                        

    print(f"complete save {saved} face image at {mode} folder ")
    return saved


for mode in ['train', 'test']:
#    os.makedirs(os.path.join(crop_folder, mode))
    os.makedirs(os.path.join(crop_folder, mode, 'live'))
    os.makedirs(os.path.join(crop_folder, mode, 'spoof'))
print('complete make directory!')


# load helper json file to read data
helper_train_json = json.load(open(helper_train_json_path))
helper_test_json = json.load(open(helper_test_json_path))

number_valid_sample = detect_face_and_crop(helper_test_json, 'test', crop_folder)

number_train_sample = detect_face_and_crop(helper_train_json, 'train', crop_folder)

# number_test_sample = detect_face_and_crop(helper_test_json, 'valid', crop_folder)

with open('number_sample.txt','w') as f:
    f.write('%d' %number_train_sample)
    f.write('\n%d' % number_valid_sample)


print('complete make croped data for training')
