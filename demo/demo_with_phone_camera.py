import numpy as np
import os
import cv2
from tensorflow.keras.models import load_model 
from model_zoo import build_efficient_net_b4, build_efficient_b7
import tensorflow as tf
from tensorflow import keras


face_threshold = 0.7

# loading face detection model
detector = '/home/duong/project/pyimage_research/version2_change_data/face_detector'
print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


model = build_efficient_net_b4(224, 2)

path_to_weight = '/home/duong/project/pyimage_research/version2_change_data/b4_result_200k_celebphoto/checkpoint/cp-01.ckpt'

model.load_weights(path_to_weight)


labels = ['live', 'spoof']


url = 'http://10.90.31.90:8080/video'

cap = cv2.VideoCapture(url)


while(True):
    ret, frame = cap.read()

    ## show video only
    # if frame is not None:
    #     cv2.imshow('frame',frame)

    ## liveness detection
    if ret == True:

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            
            if confidence > face_threshold:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
            
                if startX <= w and endX <= w and startY <= h and endY <= h:

                    try: 
                        face = frame[startY:endY, startX:endX]
                        face = cv2.resize(face, (224, 224))
                        face = face.astype("float")
                        face = np.array(face)
                        face = np.expand_dims(face, axis=0)

                        # pass the face ROI through the trained liveness detector
                        # model to determine if the face is "real" or "fake"
                        preds = model.predict(face)[0]
                        j = np.argmax(preds)
                        label = labels[j]
                        # print(preds)
                        # print(j)
                        # print(label)

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
                        print('something went wrong')
    

        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break


    q = cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()