import numpy as np
import os, datetime
import cv2
from tensorflow.keras.models import load_model 
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
from PIL import Image


# loading face detection model
detector = '/home/duong/project/pyimage_research/code/version2_change_data/face_detector'
print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

labels = ['live', 'spoof']

def demo_camera(model, face_threshold, spoof_threshold):
    # define a video capture object
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        # grab the frame dimensions and convert it to a blob
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            
            confidence = detections[0, 0, i, 2]
            if confidence > face_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
            
                if startX <= w and endX <= w and startY <= h and endY <= h:
                    face = frame[startY:endY, startX:endX]
                    print(face.shape)
                    if face.shape[0] + face.shape[1] > 150:
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
                        # j = np.argmax(preds)
                        if preds[1] > spoof_threshold:
                            j = 1
                        else:
                            j = 0
                        if w < 800:
                            font_rate = 1
                            font_size = 1
                        else:
                            font_rate = 4
                            font_size = 3
                        if j == 0:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
                            _label = "Live-Score: {:.4f}".format(preds[1])
                            cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, font_rate, (0, 255, 0), font_size)
                        else:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
                            _label = "Spoof-Score: {:.4f}".format(preds[1])
                            cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, font_rate, (0, 0, 255), font_size)


        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # load full model (.h5 file)
    model_path = '/home/duong/project/pyimage_research/result_model/version_2/fail_augmentation/result_new_b0_ver4/cp_02.h5'
    model = load_model(model_path)

    demo_camera(model, 0.5, 0.19)
