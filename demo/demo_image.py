import numpy as np
import os, datetime
import cv2
from tensorflow.keras.models import load_model 
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time

def demo_image(img_path):
    """
    this function is detect one image
    """
    if not os.path.exists(img_path):
        print("The path does not exists, please check again!")
        return None
    try:
        frame = cv2.imread(img_path)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in tqdm(range(0, detections.shape[2])):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            
            if confidence > face_threshold:

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
            
                if startX <= w and endX <= w and startY <= h and endY <= h:

                    try: 

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
                        # print(preds)
                        # print(j)
                        # print(label)

                        # draw the label and bounding box on the frame
                        label = "{}: {:.4f}".format(label, preds[j])
                        
                        if j == 0:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
                            _label = "Liveness: {:.4f}".format(preds[j])
                            cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                        else:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
                            _label = "Fake: {:.4f}".format(preds[j])
                            cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                        # end = datetime.datetime.now()
                        # delta = str(end-start)
                        # print("\n Time taken (h/m/s): %s" %delta[:7])
                        
                    except:
                        print('something went wrong')
    
        frame = cv2.resize(frame, (640,480))

        img_name = img_path.split("/")[-1]
        # Display the resulting frame
        cv2.imshow(img_name, frame)
            
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    except:
        print("Cannot read image")


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
    model_path = '/home/duong/project/pyimage_research/code/version2_change_data/efficient_b0.h5'
    model = load_model(model_path)

    img_path = '/home/duong/Desktop/test_spoof_card/31.jpg'
    demo_image(model, img_path)

