import os
import json
import sys
import cv2
import numpy as np
from config import helper_train_json_path, helper_test_json_path, detector, crop_folder
from tqdm import tqdm


print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

os.remove('broke_image.txt')

def detect_face_and_crop(dictionary_from_json, mode , folder_output):
    saved = 0
    for key, value in tqdm(dictionary_from_json.items()):
        check_exist = 0
        partitions = key.split("/")
        if value[43] == 0:
            p = os.path.join(folder_output, mode,'live', partitions[-1])
            if os.path.exists(p):
                check_exist = 1
        if value[43] == 1:
            p = os.path.join(folder_output, mode,'spoof', partitions[-1])
            if os.path.exists(p):
                check_exist = 1
        if check_exist == 0:
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
                    i = np.argmax(detections[0, 0, :, 2])
                    confidence = detections[0, 0, i, 2]
                    # ensure that the detection with the largest probability also
                    # means our minimum probability test (thus helping filter out
                    # weak detections)
                    if confidence > 0.5:
                        # compute the (x, y)-coordinates of the bounding box for
                        # the face and extract the face ROI
                        try:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            face = frame[startY:endY, startX:endX]
                            face = cv2.resize(face, (224,224))
                            # write the frame to disk
                            # p = key.replace("Data", "crop")
                            # if not os.path.exists(p[:len(p) - len(p.split("/")[-1])]):
                            #     os.makedirs(p[:len(p) - len(p.split("/")[-1])])
                            # cv2.imwrite(p, face)
                            if value[43] == 0:
                                p = os.path.join(folder_output, mode,'live', partitions[-1])
                                if not cv2.imwrite(p, face):
                                    raise Exception("Could not write image")
                                saved += 1
                            if value[43] == 1:
                                p = os.path.join(folder_output, mode,'spoof', partitions[-1])
                                if not cv2.imwrite(p, face):
                                    raise Exception("Could not write image")
                                saved += 1
                        except:
                            with open('broke_image.txt', 'a') as f:
                                f.write('\n')
                                f.write(key)
            except:
                with open('broke_image.txt', 'a') as f:
                    f.write('\n')
                    f.write(key)                

    print(f"complete save {saved} face image at {mode} folder ")
    return saved


# for mode in ['train', 'test', 'valid']:
# #    os.makedirs(os.path.join(crop_folder, mode))
#     os.makedirs(os.path.join(crop_folder, mode, 'live'))
#     os.makedirs(os.path.join(crop_folder, mode, 'spoof'))
# print('complete make directory!')


# load helper json file to read data
helper_train_json = json.load(open(helper_train_json_path))
helper_test_json = json.load(open(helper_test_json_path))

number_valid_sample = detect_face_and_crop(helper_test_json, 'test', crop_folder)

number_train_sample = detect_face_and_crop(helper_train_json, 'train', crop_folder)

# number_test_sample = detect_face_and_crop(helper_test_json, 'valid', crop_folder)

with open('number_sample.txt','w') as f:
    f.write('%d' %number_train_sample)
    f.write('\n%d' % number_valid_sample)
    f.write('\n%d' % number_valid_sample)

print('done')
