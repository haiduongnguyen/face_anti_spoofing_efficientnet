from matplotlib.pyplot import axis
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
from losses import binary_focal_loss, categorical_focal_loss

import dill
custom_object = {'binary_focal_loss_fixed': dill.loads(dill.dumps(binary_focal_loss(gamma=2., alpha=.25))),
                 'categorical_focal_loss_fixed': dill.loads(dill.dumps(categorical_focal_loss(gamma=2., alpha=[[.25, .25, .25]]))),
                 'categorical_focal_loss': categorical_focal_loss,
                 'binary_focal_loss': binary_focal_loss}


def eval(model_name, model_path, index):
    checkpoint_path = os.path.join(model_path, index)
    if os.path.exists(checkpoint_path):
      model = load_model(checkpoint_path, custom_objects=custom_object)

      result_folder = work_place + '/result_' + model_name 
      # saver = tf.train.Saver()
      # saver = tf.compat.v1.train.Saver()
      # sess = keras.backend.get_session()
      # saver.restore(sess, result_folder + '/keras_session')

      result_test_folder = result_folder + '/test' + '_' + index[:-3]
      if not os.path.isdir(result_test_folder):
          os.makedirs(result_test_folder)

      result_txt = result_test_folder + '/result_test.txt'
      score_txt = result_test_folder + '/score_prediction.txt'
      wrong_live_txt = result_test_folder + '/wrong_live_sample.txt'
      wrong_spoof_txt = result_test_folder + '/wrong_spoof_sample.txt'

      scores = []



      with open(result_txt, 'w') as f:
        f.close()
      path_live = os.path.join(crop_data_test, 'live')
      path_spoof = os.path.join(crop_data_test, 'spoof')
      print("model name and version is: " + model_path, file=open(result_txt, 'a'))
      print("live test folder at: " + path_live, file=open(result_txt, 'a'))
      print("spoof test folder at: " + path_spoof, file=open(result_txt, 'a'))


      count_live = 0
      for image_name in tqdm(os.listdir(path_live)):
        image = cv2.imread(os.path.join(path_live, image_name))
        image = cv2.resize(image, (image_size,image_size))
        # image = np.array(image, 'float32')
        image = np.expand_dims(image, 0)
        score = model.predict(image)
        scores.append(score)
        count_live += 1
      print(count_live)



      count_spoof = 0
      for image_name in tqdm(os.listdir(path_spoof)):
        image = cv2.imread(os.path.join(path_spoof, image_name))
        image = cv2.resize(image, (image_size,image_size))
        # image = np.array(image, 'float32')
        image = np.expand_dims(image, 0)
        score = model.predict(image)
        scores.append(score)
        count_spoof += 1
      print(count_spoof)

      scores = np.array(scores)
      print("prediction scores have shape: " + str(scores.shape), file=open(result_txt, 'a'))

      list_live = [0]*count_live
      list_spoof = [1]*count_spoof
      labels = list_live + list_spoof
      labels = np.array(labels, dtype=np.float32)
      print("labels have shape: " + str(labels.shape), file=open(result_txt, 'a'))
      # labels = np.array(labels)
      # labels = tf.keras.utils.to_categorical( labels, num_classes=2, dtype='float32')
      # labels = np.array(labels)

      live_score = np.array(scores[:,0,0], dtype=np.float32)
      # result_live = cal_metric(labels, live_score)
      # print('eer live is : ', result_live[0] , file=open('result_test.txt', 'a'))
      # print('tpr live is : ', result_live[1] , file=open('result_test.txt', 'a'))
      # print('auc live is : ', result_live[2] , file=open('result_test.txt', 'a'))

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

        with open(wrong_spoof_txt, 'w') as f:
          for item in wrong_spoof_list:
              f.write("%s\n" % item)

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

        with open(wrong_live_txt, 'w') as f:
            for item in wrong_live_list:
                f.write("%s\n" % item)

        avg_wrong_rate = round((wrong_live + wrong_spoof)/(count_live + count_spoof), 4)
        print(f"the average wrong rate of model is: {avg_wrong_rate}", file=open(result_txt, 'a'))
        try:
          valid_datagen = ImageDataGenerator()   
          validation_dir = crop_data_test
          validation_generator = valid_datagen.flow_from_directory(
                  validation_dir,
                  target_size=(image_size, image_size),
                  batch_size=1,
                  class_mode='categorical')
          # print(model.evaluate())
          opt_adam = keras.optimizers.Adam(lr=INIT_LR)
          model.compile(loss="categorical_crossentropy", optimizer=opt_adam, metrics=['binary_accuracy', 'categorical_accuracy'])
          a = model.evaluate(validation_generator, batch_size=1)
          print(a,file=open(result_txt, 'a'))
        except:
          pass

      else:
        print('something went wrong, check again')
    else:
      print("No checkpoint at the path, check again!")


if __name__ == '__main__':
  # model_name = 'new_b0_ver0'
  # model_path = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet' + '/result_' + model_name + '/train/checkpoint'
  # index_checkpoint = ['cp_02.h5' , 'cp_03.h5']
  # for index in index_checkpoint:
  #   eval(model_name, model_path, index)    

  # model_name = 'new_b0_add_convolutional_layer'
  # model_path = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet' + '/result_' + model_name + '/train/checkpoint'
  # index_checkpoint = ['cp_03.h5' , 'cp_05.h5']
  # for index in index_checkpoint:
  #   eval(model_name, model_path, index)  

  model_name = 'new_b0_ver3'
  model_path = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet' + '/result_' + model_name + '/train/checkpoint'
  index_checkpoint = ['cp_06.h5' , 'cp_08.h5']
  for index in index_checkpoint:
    eval(model_name, model_path, index)  