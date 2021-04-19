import numpy as np
import argparse
import cv2
import os
import glob


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--detector", type=str, required=True,
# 	help="path to OpenCV's deep learning face detector")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
#     help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

args = {}
# path to model detect face 
args["detector"] = "./face_detector"
# folder of output 
args["output"] = '/home/duong/project/pyimage_research/version2_change_data/test_image/crop_data/test'


print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# open a pointer to the video file stream and initialize the total
# number of frames read and saved thus far

mother_test_path = '/home/duong/project/pyimage_research/version2_change_data/test_image/rotated_data/test'
list_folder = os.listdir(mother_test_path)

count = 0

for fol in list_folder:
    if len(fol) <= 0:
        pass
    else:
        live_path = os.path.join(mother_test_path, fol, "live")
        spoof_path = os.path.join(mother_test_path, fol, "spoof")

        if os.path.isdir(live_path):
            saved = 0
            live_list = glob.glob1(live_path, "*.png")
            for image in live_list:
                path_to_image = os.path.join(live_path, image)
                frame = cv2.imread(path_to_image)

                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))
                # pass the blob through the network and obtain the detections and
                # predictions
                net.setInput(blob)
                detections = net.forward()
                # ensure at least one face was found
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
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = frame[startY:endY, startX:endX]
                        # write the frame to disk
                        p = args["output"] + '/live' + '/' + image[:-4] + '.jpg'
                        cv2.imwrite(p, face)
                        saved += 1
            count += saved
            print(f"complete cut {saved} face at live folder {fol}")

        if os.path.isdir(spoof_path):
            saved = 0
            spoof_list = glob.glob1(spoof_path, "*.png")
            for image in spoof_list:
                path_to_image = os.path.join(spoof_path, image)
                frame = cv2.imread(path_to_image)

                (h, w) = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))

                # pass the blob through the network and obtain the detections and
                # predictions
                net.setInput(blob)
                detections = net.forward()
                # ensure at least one face was found
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
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = frame[startY:endY, startX:endX]
                        # write the frame to disk
                        p = args["output"] + '/spoof' + '/' + image[:-4] + '.jpg'
                        cv2.imwrite(p, face)
                        saved += 1
            count += saved
            print(f"complete cut {saved} face at spoof folder {fol}")
        

print("complete cut face of " + str(count)+ " images")
