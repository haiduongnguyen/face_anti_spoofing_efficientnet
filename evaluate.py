import numpy as np
import keras
import tensorflow as tf
import os, datetime
import numpy as np
import cv2
# if read image from matolotlib, what will happen
#from matplotlib.image import imread
# my packages
from config import *
from model_zoo import *
from eer_calculation import cal_metric
from keras.models import load_model

model = load_model(model_name)
scores = []

path_live = os.path.join(crop_data_test, 'live')
path_spoof = os.path.join(crop_data_test, 'spoof')
print(path_live, file=open('result_test.txt', 'a'))
print(path_spoof, file=open('result_test.txt', 'a'))

count_live = 0
for image_path in os.listdir(path_live):
  image = cv2. imread(os.path.join(path_live, image_path))
  image = np.array(image, 'float32')/255
  image = np.expand_dims(image, 0)
  score = model.predict(image)
  scores.append(score)
  count_live += 1



count_spoof = 0
for image_path in os.listdir(path_spoof):
  image = cv2.imread(os.path.join(path_spoof, image_path))
  image = np.array(image, 'float32')/255
  image = np.expand_dims(image, 0)
  score = model.predict(image)
  scores.append(score)
  count_spoof += 1

scores = np.array(scores)
print("prediction scores have shape: ", scores.shape)

list_live = [0]*count_live
list_spoof = [1]*count_spoof
labels = list_live + list_spoof
labels = np.array(labels)
print("labels have shape: ", labels.shape)
# labels = np.array(labels)
# labels = tf.keras.utils.to_categorical( labels, num_classes=2, dtype='float32')
# labels = np.array(labels)


live_score = np.array(scores[:,0,0])
result_live = cal_metric(labels, live_score)
print('eer live is : ', result_live[0] , file=open('result_test.txt', 'a'))
print('tpr live is : ', result_live[1] , file=open('result_test.txt', 'a'))
print('auc live is : ', result_live[2] , file=open('result_test.txt', 'a'))

spoof_score = np.array(scores[:,0,1])
result_spoof = cal_metric(labels, spoof_score)
print('eer spoof is : ', result_spoof[0] , file=open('result_test.txt', 'a'))
print('tpr spoof is : ', result_spoof[1] , file=open('result_test.txt', 'a'))
print('auc spoof is : ', result_spoof[2] , file=open('result_test.txt', 'a'))


print('test set has number live sample : ' + str(count_live), file=open('result_test.txt', 'a'))
print('test set has number spoof sample : ' + str(count_spoof), file=open('result_test.txt', 'a'))
# calculate apcer, bpcer
predict_score = np.stack([live_score, spoof_score], axis=1)

prediction = np.round(predict_score)

test_len = len(prediction)

if predict_score.shape[0] == labels.shape[0]:
  predict_live = 0
  wrong_spoof = 0
  for i in range(test_len):
    if prediction[i][0] == 1:
      predict_live += 1
      if labels[i] == 0 :
        wrong_spoof += 1
      # wrong_list.append(i)
  if predict_live == 0:
    print('No prediction is live', file=open('result_test.txt', 'a'))
    wrong_rate = 0
  else:
    print(f"number of live samples is predicted as spoof is {wrong_spoof}", file=open('result_test.txt', 'a'))
    wrong_rate = round(wrong_spoof/predict_live, 4)
  print(f"model predict number of sample as live : {predict_live}", file=open('result_test.txt', 'a'))
  print(f"model has wrong live rate (BPCER) = {wrong_rate} ", file=open('result_test.txt', 'a'))


  predict_spoof = 0
  wrong_live = 0
  for i in range(test_len):
    if prediction[i][0] == 0:
      predict_spoof += 1
      if labels[i] == 1 :
        wrong_live += 1
      # wrong_list.append(i)
  if predict_spoof == 0:
    print('No prediction is spoof')
    wrong_rate = 0
  else:
    print(f"number of spoof samples is predicted as live is {wrong_live}", file=open('result_test.txt', 'a'))
    wrong_rate = round(wrong_live/predict_spoof, 4)
  print(f"model predict number of sample as spoof : {predict_spoof}", file=open('result_test.txt', 'a'))
  print(f"model has wrong spoof rate (APCER) = {wrong_rate}", file=open('result_test.txt', 'a'))

  # with open(folder_save_model + '/' + model_name + '_wrong_sample.txt', 'w') as f:
  #     for item in wrong_list:
  #         f.write("%s\n" % item)
else:
  print('something went wrong, check again')
