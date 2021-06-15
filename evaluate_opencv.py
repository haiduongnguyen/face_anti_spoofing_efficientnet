"""
The problem is evaluate give different from predict one image
I found a problem to this evaluate cause of 
    cv2 read image differ from tensorflow read image
    cv2.resize is different from tf.image.resize in flow_from_directory function
so i have write a new script that use predict from flow_directory function and see the eer again
I waste 3 weeks to recognize a this :( so hope can be useful for other
"""
import numpy as np
import keras
import tensorflow as tf
import os, datetime
import numpy as np
import cv2
# if read image from matolotlib, what will happen
#from matplotlib.image import imread
# my packages
from keras.preprocessing.image import ImageDataGenerator
from config import *
from model_zoo import *
from eer_calculation import cal_metric
from keras.models import load_model
from tqdm import tqdm
from model_zoo import *
from PIL import Image


def eval_opencv(all_checkpoint_path, index_cp, result_folder):

    checkpoint_path = os.path.join(all_checkpoint_path, index_cp)
    if not os.path.exists(checkpoint_path):
      print("No checkpoint at the path, check again!")
      return -1

    model = load_model(checkpoint_path)

    input_model = model.input_shape
    width , height = input_model[1], input_model[2]

    result_test_folder = result_folder + '/test' + '_opencv_' + index_cp[:-3]
    if not os.path.isdir(result_test_folder):
        os.makedirs(result_test_folder)

    result_txt = result_test_folder + '/result_test.txt'
    score_txt = result_test_folder + '/score_prediction.txt'
    wrong_sample_txt = result_test_folder + '/wrong_sample.txt'

    path_live = os.path.join(crop_data_test, 'live')
    path_spoof = os.path.join(crop_data_test, 'spoof')
    print("model name and version is: " + checkpoint_path, file=open(result_txt, 'a'))
    print("live test folder at: " + path_live, file=open(result_txt, 'a'))
    print("spoof test folder at: " + path_spoof, file=open(result_txt, 'a'))

    scores = []
    count_live = 0
    live_image_list = os.listdir(path_live)
    for image_name in tqdm(live_image_list):
      image = cv2.imread(os.path.join(path_live, image_name))
      
      # use cv2.resize
      image = cv2.resize(image, (width, height))
      image = np.expand_dims(image, 0)
      score = model.predict(image)
      scores.append(score)
      count_live += 1
    print('number sample live : ', count_live)

    count_spoof = 0
    spoof_image_list = os.listdir(path_spoof)
    image_list = live_image_list + spoof_image_list
    for image_name in tqdm(spoof_image_list):
      image = cv2.imread(os.path.join(path_spoof, image_name))
    
      image = cv2.resize(image, (width, height))
      image = np.array(image, 'float32')
      image = np.expand_dims(image, 0)
      
      score = model.predict(image)
      scores.append(score)
      count_spoof += 1
    print('number sample spoof', count_spoof)

    scores = np.array(scores)
    print("prediction scores have shape: " + str(scores.shape), file=open(result_txt, 'a'))

    list_live = [0]*count_live
    list_spoof = [1]*count_spoof
    labels = list_live + list_spoof
    labels = np.array(labels, dtype=np.float32)
    print("labels have shape: " + str(labels.shape), file=open(result_txt, 'a'))

    live_score = np.array(scores[:,0,0], dtype=np.float32)
    spoof_score = np.array(scores[:,0,1], dtype=np.float32)

    result_spoof = cal_metric(labels, spoof_score)
    print('eer spoof is : ' + str(result_spoof[0]) , file=open(result_txt, 'a'))
    print('tpr spoof is : ' + str(result_spoof[1]) , file=open(result_txt, 'a'))
    print('auc spoof is : ' + str(result_spoof[2]) , file=open(result_txt, 'a'))
    print('threshold for eer is : ' + str(result_spoof[4]) , file=open(result_txt, 'a'))

    print('test set has number live sample : ' + str(count_live), file=open(result_txt, 'a'))
    print('test set has number spoof sample : ' + str(count_spoof), file=open(result_txt, 'a'))
    # calculate apcer, bpcer
    predict_score = np.stack([live_score, spoof_score], axis=1)
    with open(score_txt, 'w') as f:
        for item in spoof_score:
            f.write("%s\n" % item)

    prediction = [1]*(spoof_score.shape[0])
    for i in range(spoof_score.shape[0]):
        if spoof_score[i] < result_spoof[4]:
            prediction[i] = 0

    # prediction = np.argmax(predict_score, axis=1)

    test_len = len(prediction)

    wrong_spoof_list = []
    if predict_score.shape[0] == labels.shape[0]:
      predict_live = 0
      wrong_spoof = 0
      for i in range(test_len):
        if prediction[i] == 0:
          predict_live += 1
          if labels[i] == 1 :
            wrong_spoof += 1
            wrong_spoof_list.append(i)
      if predict_live == 0:
        print('No prediction is live', file=open(result_txt, 'a'))
        wrong_rate = 0
      else:
        print(f"number of spoof samples is predicted as live is {wrong_spoof}", file=open(result_txt, 'a'))
        wrong_rate = round(wrong_spoof/count_live, 4)
      print(f"model predict number of sample as live : {predict_live}", file=open(result_txt, 'a'))
      print(f"model has wrong live rate (BPCER) = {wrong_rate} ", file=open(result_txt, 'a'))

      with open(wrong_sample_txt, 'w') as f:
        for img_index in wrong_spoof_list:
          img_path = os.path.join(path_spoof, image_list[img_index])
          f.write("%s\n" % img_path)

      predict_spoof = 0
      wrong_live = 0
      wrong_live_list = []
      for i in range(test_len):
        if prediction[i] == 1:
          predict_spoof += 1
          if labels[i] == 0 :
            wrong_live += 1
            wrong_live_list.append(i)
      if predict_spoof == 0:
        print('No prediction is spoof')
        wrong_rate = 0
      else:
        print(f"number of live samples is predicted as spoof is {wrong_live}", file=open(result_txt, 'a'))
        wrong_rate = round(wrong_live/count_spoof, 4)
      print(f"model predict number of sample as spoof : {predict_spoof}", file=open(result_txt, 'a'))
      print(f"model has wrong spoof rate (APCER) = {wrong_rate}", file=open(result_txt, 'a'))

      with open(wrong_sample_txt, 'a') as f:
          for img_index in wrong_live_list:
            img_path = os.path.join(path_live, image_list[img_index])
            f.write("%s\n" % img_path)
      
      avg_wrong_rate = round((wrong_live + wrong_spoof)/(count_live + count_spoof), 4)
      print(f"the average wrong rate of model is: {avg_wrong_rate}", file=open(result_txt, 'a'))

    else:
      print('something went wrong, check again')


if __name__ == '__main__':

  # change model name
  model_name = 'b0_ver_3'
  result_folder = result_all_model + '/' + model_name 
  all_checkpoint_path = result_folder + '/train/checkpoint_' + model_name

  # change list of cp will be evaluated
  index_cp_list = ['cp_06.h5' , 'cp_08.h5']
  for index_cp in index_cp_list:
    eval_opencv(all_checkpoint_path, index_cp, result_folder)  
