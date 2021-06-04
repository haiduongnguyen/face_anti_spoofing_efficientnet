import numpy as np
import os, datetime
import cv2
from tensorflow.keras.models import load_model 
from model_zoo import *
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time

def demo_camera():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            print(i)
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > face_threshold:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
                try: 
                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (224, 224))
                    face = face.astype("float") 
                    face = np.array(face)
                    face = np.expand_dims(face, axis=0)

                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"
                    preds = model.predict(face)[0][1]
                    if preds > 0.11762203:
                        j = 1
                    else:
                        j = 0
                    # j = np.argmax(preds)
                    label = labels[j]
                    print(preds)
                    print(j)
                    print(label)

                    # draw the label and bounding box on the frame
                    label = "{}: {:.4f}".format(label, preds[j])
                    
                    if j == 0:
                        cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 255, 0), 2)
                        _label = "Liveness: {:.4f}".format(preds[j])
                        cv2.putText(frame, _label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                        _label = "Fake: {:.4f}".format(preds[j])
                        cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except:
                    pass


        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_threshold = 0.7

    # loading face detection model
    detector = '/home/duong/project/pyimage_research/version2_change_data/face_detector'
    print("[INFO] loading face detector...")
    protoPath = os.path.join(detector, "deploy.prototxt")
    modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


    labels = ['live', 'spoof']
    # load full model (.h5 file)
    model_path = '/home/duong/project/pyimage_research/version2_change_data/result_new_b4_ver01/cp_04.h5'
    model = load_model(model_path)
