import cv2
import os
import numpy as np
from tqdm import tqdm

detector = './face_detector'

face_threshold = 0.7

model_name = './origin_code/liveness.model'

print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


with open('broken_image.txt', 'w') as f:
    f.close()

def detect_face_and_crop(input_folder, mode , output_folder):
    saved = 0
    for type in ['live', 'spoof']:
        folder_path = os.path.join(input_folder, mode, type)

        for name_image in tqdm(os.listdir(folder_path)):
            try:
                frame = cv2.imread(os.path.join(folder_path, name_image))

                (h, w) = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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
                    # if confidence > 0.5:
                    #     # compute the (x, y)-coordinates of the bounding box for
                    #     # the face and extract the face ROI
                    #     try:
                    #         box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    #         (startX, startY, endX, endY) = box.astype("int")
                    #         face = frame[startY:endY, startX:endX]
                    #         face = cv2.resize(face, (224,224))
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

                            p = os.path.join(output_folder, mode, type, name_image)
                            if 'jpeg' in p:
                                p = p.replace('jpeg', 'jpg')
                            if not cv2.imwrite(p, face):
                                raise Exception("Could not write image")
                            saved += 1
            except:
                print(name_image, file=open('broken_image.txt', 'a'))

    print(f"complete save {saved} face image at {mode} folder ")


raw_folder = '/home/duong/project/pyimage_research/version2_change_data/small_data_to_test/raw_data'

crop_folder = '/home/duong/project/pyimage_research/version2_change_data/small_data_to_test/crop_data'

def main():

    detect_face_and_crop(raw_folder, 'test', crop_folder)

    detect_face_and_crop(raw_folder, 'train', crop_folder)




if __name__ == '__main__':
    main()
