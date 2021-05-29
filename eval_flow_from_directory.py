"""
The problem is evaluate give different from predict one image
I found a problem to this evaluate cause of cv2.resize is different from tf.image.resize in flow_from_directory function
so i have write a new script that use predict from flow_directory function and see the eer again
I waste 3 weeks to recognize a this :( so hope can be useful for other
"""

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

        result_test_folder = result_folder + '/test_flow_from_directory_'  + index[:-3]
        if not os.path.isdir(result_test_folder):
            os.makedirs(result_test_folder)

        result_txt = result_test_folder + '/result_test.txt'
        with open(result_txt, 'w') as f:
            f.close()
        spoof_score_txt = result_test_folder + '/score_prediction.txt'
        with open(spoof_score_txt, 'w') as f:
            f.close()
        wrong_live_txt = result_test_folder + '/wrong_live_sample.txt'
        with open(wrong_live_txt, 'w') as f:
            f.close()
        wrong_spoof_txt = result_test_folder + '/wrong_spoof_sample.txt'
        with open(wrong_spoof_txt, 'w') as f:
            f.close()


        valid_datagen = ImageDataGenerator()   
        validation_dir = crop_data_test
        validation_generator = valid_datagen.flow_from_directory(
                validation_dir,
                target_size=(image_size, image_size),
                batch_size=1,
                shuffle= False,
                class_mode='categorical')

        model.compile(loss="categorical_crossentropy",  metrics=['binary_accuracy', 'categorical_accuracy'])
        a = model.evaluate(validation_generator, batch_size=1)
        print(a,file=open(result_txt, 'a'))

        filenames = validation_generator.filenames
        nb_samples = len(filenames)

        prediction = model.predict(validation_generator, steps = nb_samples)

        spoof_score = np.array(prediction)[:,1]
        print("prediction scores have shape: " + str(spoof_score.shape), file=open(result_txt, 'a'))


        path_live = os.path.join(crop_data_test, 'live')
        path_spoof = os.path.join(crop_data_test, 'spoof')
        count_live = len(os.listdir(path_live))
        count_spoof = len(os.listdir(path_spoof))
        list_live = [0]*count_live
        list_spoof = [1]*count_spoof
        labels = list_live + list_spoof
        labels = np.array(labels, dtype=np.float32)
        print("labels have shape: " + str(labels.shape), file=open(result_txt, 'a'))


        result_spoof = cal_metric(labels, spoof_score)
        print('eer spoof is : ' + str(result_spoof[0]) , file=open(result_txt, 'a'))
        print('tpr spoof is : ' + str(result_spoof[1]) , file=open(result_txt, 'a'))
        print('auc spoof is : ' + str(result_spoof[2]) , file=open(result_txt, 'a'))
        print('threshold for eer is : ' + str(result_spoof[4]) , file=open(result_txt, 'a'))



        print('test set has number live sample : ' + str(count_live), file=open(result_txt, 'a'))
        print('test set has number spoof sample : ' + str(count_spoof), file=open(result_txt, 'a'))

        with open(spoof_score_txt, 'w') as f:
            for item in spoof_score:
                f.write("%s\n" % item)

        
        prediction_class = [1]*(spoof_score.shape[0])
        for i in range(spoof_score.shape[0]):
            if spoof_score[i] < result_spoof[4]:
                prediction_class[i] = 0

      # prediction = np.argmax(predict_score, axis=1)


        test_len = len(prediction)

        wrong_spoof_list = []
        predict_live = 0
        wrong_spoof = 0
        for i in range(test_len):
            if prediction_class[i] == 0:
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
            if prediction_class[i] == 1:
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

#   model_name = 'new_b0_ver3'
#   model_path = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet' + '/result_' + model_name + '/train/checkpoint'
#   index_checkpoint = ['cp_06.h5' , 'cp_08.h5']
#   for index in index_checkpoint:
#     eval(model_name, model_path, index)  

  model_name = 'new_b0_ver2'
  model_path = '/home/duongnh/liveness_detection_efficienetb4_20210515_ver02/face_anti_spoofing_efficientnet' + '/result_' + model_name + '/train/checkpoint'
  index_checkpoint = ['cp_03.h5' , 'cp_08.h5']
  for index in index_checkpoint:
    eval(model_name, model_path, index)  