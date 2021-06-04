import numpy as np
import os
import cv2
from keras.models import load_model as load_model


detector = './face_detector'

face_threshold = 0.7


# loading models
print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def use_camera():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    while True:

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        frame = cv2.resize(frame, (640,480))

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        print(type(detections))
        print(detections.shape)
        count_face = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > face_threshold:
                count_face += 1

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                if startX <= w and endX <= w and startY <= h and endY <= h:
                    # filter out weak detections
                    face = frame[startY:endY, startX:endX]
                    if 0 not in face.shape:
                        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        print(count_face)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


def use_image(img_path):
    # define a video capture object
    img = cv2.imread(img_path)

    frame = cv2.resize(img, (480,640))

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    print(type(detections))
    print(detections.shape)
    count_face = 0
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > face_threshold:
            count_face += 1

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            if startX <= w and endX <= w and startY <= h and endY <= h:
                # filter out weak detections
                face = frame[startY:endY, startX:endX]
                if 0 not in face.shape:
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

    print(count_face)
    # Display the resulting frame
    cv2.imshow('img', img)
    cv2.imshow('frame', frame)

    cv2.waitKey(0) 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img_path = '/home/duong/project/pyimage_research/version2_change_data/image_to_test/nam_1.jpg'
    use_image(img_path)
