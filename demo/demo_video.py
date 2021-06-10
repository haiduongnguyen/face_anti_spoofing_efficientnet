import numpy as np
import os, datetime
import cv2
from tensorflow.keras.models import load_model 
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import time
from PIL import Image


class Demo_video():
    def __init__(self, video_input, video_output, save_output, face_detect_model, face_threshold, spoof_detect_model, spoof_threshold, labels) :
        self.video_input = video_input
        self.video_output = video_output
        self.save_output = save_output
        self.face_detect_model = face_detect_model 
        self.face_threshold = face_threshold
        self.spoof_detect_model = spoof_detect_model
        self.spoof_threshold = spoof_threshold
        self.labels = labels



    def demo_video(self):
        # read video
        vid = cv2.VideoCapture(self.video_input)
        # out = cv2.VideoWriter('1_result.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (480,640))
        if self.save_output == True:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(self.video_output, fourcc, 30, (640,480))
        ret = True
        while ret:
            ret , frame = vid.read()

            if ret == True:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # grab the frame dimensions and convert it to a blob
                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame_rgb, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                self.face_detect_model.setInput(blob)
                detections = self.face_detect_model.forward()

                for i in range(0, detections.shape[2]):
                    # extract the confidence (i.e., probability) associated with the
                    # prediction
                    confidence = detections[0, 0, i, 2]
                    
                    if confidence > self.face_threshold:

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                    
                        if startX <= w and endX <= w and startY <= h and endY <= h:

                            # try: 
                            face = frame_rgb[startY:endY, startX:endX]

                            # resize using cv2
                            input_model = self.spoof_detect_model.input_shape
                            width , height = input_model[1], input_model[2]
                            face = cv2.resize(face, (width, height))
                            face = face.astype("float")
                            face = np.array(face)
                            face = np.expand_dims(face, axis=0)
                            
                            # pass the face ROI through the trained liveness detector
                            # model to determine if the face is "real" or "fake"

                            preds = self.spoof_detect_model.predict(face)[0]
                            if preds[1] > self.spoof_threshold:
                                j = 1
                            else:
                                j = 0

                            # # draw the label and bounding box on the frame
                            # label = "{}: {:.4f}".format(label, preds[1])
                            
                            if j == 0:
                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 3)
                                _label = "Score: {:.4f}".format(preds[1])
                                cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                            else:
                                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 3)
                                _label = "Score: {:.4f}".format(preds[1])
                                cv2.putText(frame, _label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                            # except:
                            #     print('something went wrong')
            
                frame = cv2.resize(frame, (640,480))
                if self.save_output == True:
                    out.write(frame)
                # Display the resulting frame
                cv2.imshow('frame', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        

        # After the loop release the cap object
        vid.release()
        if self.save_output == True:
            out.release()
        # Destroy all the windows
        cv2.destroyAllWindows()


if __name__ == "__main__":

    video_input = '/home/duong/project/pyimage_research/video/version_2/1.mp4'
    video_output = '/home/duong/project/pyimage_research/video/version_2/new_b0_ver4_cp2.mp4'
    save_output = True
    
    face_threshold = 0.7

    # loading face detection model
    detector = '/home/duong/project/pyimage_research/code/version2_change_data/face_detector'
    protoPath = os.path.join(detector, "deploy.prototxt")
    modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
    face_detect_model = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


    # load full model (.h5 file)
    model_path = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b0_ver4/cp_02.h5'
    spoof_detect_model = load_model(model_path)
    spoof_threshold = 0.191
    labels = ['live', 'spoof']

    
    a = Demo_video(video_input, video_output, save_output, face_detect_model, face_threshold, spoof_detect_model, spoof_threshold, labels)
    a.demo_video()
