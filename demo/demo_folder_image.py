import numpy as np
import os, datetime
import cv2
from tensorflow.keras.models import load_model 
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time

def demo_image(model, img_path):
    """
    this function is detect one image
    """
    if not os.path.exists(img_path):
        print("The path does not exists, please check again!")
        return None

    frame = cv2.imread(img_path)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    
    count_face = 0
    for i in range(0, detections.shape[2]):
        
        confidence = detections[0, 0, i, 2]
        
        if confidence > face_threshold:
            count_face += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
        
            if startX <= w and endX <= w and startY <= h and endY <= h:
                face = frame[startY:endY, startX:endX]
                print(face.shape)

                input_model = model.input_shape
                width , height = input_model[1], input_model[2]
                face = frame[startY:endY, startX:endX]
                face = cv2.resize(face, (width, height))

                face = face.astype("float")
                face = np.array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = labels[j]
                return j
    if count_face == 0:
        print(img_path)
    
    # frame = cv2.resize(frame, (640,480))

    # img_name = img_path.split("/")[-1]
    # # Display the resulting frame
    # cv2.imshow(img_name, frame)
        
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    return -1


def demo_folder_image(model, folder_path = '', count_live = 0, count_spoof = 0, count_error=0):
    if not os.path.exists(folder_path):
        print("Not exits folder, check again!!")
        return None
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        j = demo_image(model, img_path)
        if j == 0:
            count_live += 1
        elif j == 1:
            count_spoof += 1
        elif j == -1:
            count_error += 1
    return count_live, count_spoof, count_error



if __name__ == '__main__':
    face_threshold = 0.7

    # loading face detection model
    detector = '/home/duong/project/pyimage_research/code/version2_change_data/face_detector'
    print("[INFO] loading face detector...")
    protoPath = os.path.join(detector, "deploy.prototxt")
    modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    labels = ['live', 'spoof']
    # load full model (.h5 file)
    model_path = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b0_ver4/cp_02.h5'
    model = load_model(model_path)

    folder_path = '/home/duong/Desktop/test_spoof_card'
    count_live, count_spoof, count_error = demo_folder_image(model, folder_path)
    print('live : ', count_live)
    print('spoof : ', count_spoof)
    print('error : ', count_error)


