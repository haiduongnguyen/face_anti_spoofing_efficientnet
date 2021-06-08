import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model
import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
# from tensorflow.python.keras.backend import sign

detector = '/home/duong/project/pyimage_research/code/version2_change_data/face_detector'
print("[INFO] loading face detector...")
protoPath = os.path.join(detector, "deploy.prototxt")
modelPath = os.path.join(detector, "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# labels = ['live', 'spoof']
model_path = '/home/duong/project/pyimage_research/result_model/version_2/result_new_b0_ver4/cp_01.h5'
# model_path = '/home/duong/project/pyimage_research/version2_change_data/result_new_b0_ver2/cp_01.h5'
model = load_model(model_path)
input_model = model.input_shape
width , height = input_model[1], input_model[2]

face_threshold = 0.7
# some eer and threshold of testing on test folder
# new_b0_ver4 - cp01 --> eer = 2.5, threshold = 0.38
spoof_threshold = 0.1

#initialise GUI
top=tk.Tk()
top.geometry('1000x800')
top.title('Face liveness detection demo')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)


def classify_image(img_path):
    if not os.path.exists(img_path):
        print("The path does not exists, please check again!")
        return None

    frame = Image.open(img_path)
    print(img_path)
    frame_cv = np.asarray(frame)
    (h, w) = frame_cv.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame_cv, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    count_dict = {"live" : 0 , "spoof" : 0}
    include_face = False
    enough_size = False 
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] 
        if confidence > face_threshold:
            print(i)
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startX <= w and endX <= w and startY <= h and endY <= h:
                # try: 
                face = frame.crop(( startX, startY, endX, endY))
                include_face = True
                if width + height >= 250:
                    enough_size = True
                    face = face.resize((width, height), Image.BILINEAR)
                    face = img_to_array(face)
                    # expand dim
                    face = np.array([face])
                    # pass the face ROI through the trained liveness detector
                    # model to determine if the face is "real" or "fake"
                    spoof_score = model.predict(face)[0][1]
                    print(spoof_score)
                    if spoof_score > spoof_threshold:
                        count_dict["spoof"] += 1
                    else:
                        count_dict["live"] += 1
                
    # if no face in image, print to screen
    if not include_face:
        sign = "No face in image"
        label.configure(foreground='#011638', text=sign) 
    else:
        if not enough_size:
            sign = "face is too small"
            label.configure(foreground='#011638', text=sign) 
        else:
            sign = count_dict
            score_label = Label(top,background='#CDCDCD', font=('arial',15,'bold'))
            score_label.configure(foreground='#023184', text="score = " + str(spoof_score))
            label.configure(foreground='#011638', text=sign) 
    

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image", command=lambda: classify_image(file_path), padx=15,pady=10)
    classify_b.configure(background='#364156', foreground='white', font=('arial',10,'bold'))
    classify_b.place(relx=0.8,rely=0.4)

def show_result(file_path):
    pass

def show_result_button(file_path):
    show_result_btn = Button(top, text="Show Result", command=lambda: show_result(file_path), padx=15, pady=10)
    show_result_btn.configure(background='#364156', foreground='red', font=('arial', 10, 'bold'))
    show_result_btn.place(relx= 0.8, rely=0.5)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/1.5), (top.winfo_height()/1.5)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
        # show_result_button(file_path)
    except:
        pass

def show_upload_button():
    upload=Button(top,text="Upload an image",command=upload_image,padx=15,pady=10)
    upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    upload.pack(side=BOTTOM,pady=50)

show_upload_button()
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Liveness detection ",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()