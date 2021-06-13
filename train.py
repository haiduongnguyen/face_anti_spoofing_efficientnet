import numpy as np
import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_lattice as tfl
import os, datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tqdm import tqdm
import statistics
# my packages
from config import *
from model_zoo import *
from eer_calculation import cal_metric
from focal_losses import categorical_focal_loss
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

start = datetime.datetime.now()


class config_to_train():
    def __init__(self,model_name='b0', build_model=build_b0_gap, 
                    img_width=224, img_height=224, img_depth=3, classes=2, INIT_LR=1e-4, batch_size=8, EPOCHS=20, patience=5):
        self.model_name = model_name
        self.build_model = build_model
        self.img_width = img_width
        self.img_height = img_height
        self.img_depth = img_depth
        self.classes = classes
        self.INIT_LR = INIT_LR
        self.batch_size = batch_size
        self.EPOCHS = EPOCHS
        self.patience = patience
        self.checkpoint_dir = None
        self.result_folder = None
    
    def train(self):
        model = self.build_model(self.img_width, self.img_height, self.img_depth, self.classes)

        self.result_folder = result_all_model + '/' + self.model_name
        if not os.path.isdir(self.result_folder):
            os.makedirs(self.result_folder)
        result_train_folder = self.result_folder + '/train'
        if not os.path.isdir(result_train_folder):
            os.makedirs(result_train_folder)

        training_output_txt = result_train_folder + '/result_training_output.txt'

        train_datagen = ImageDataGenerator(rotation_range=20,horizontal_flip=True,fill_mode='nearest',brightness_range=[0.75,1.25])

        train_dir = crop_data_train
        train_generator = train_datagen.flow_from_directory(
                # This is the target directory
                train_dir,
                # All images will be resized to 224x224
                target_size=(self.img_width, self.img_height), batch_size=self.batch_size,
                class_mode='categorical', interpolation="bilinear",  seed=2021)

        # Note that the validation data should not be augmented!
        valid_datagen = ImageDataGenerator()   
        validation_dir = crop_data_test
        validation_generator = valid_datagen.flow_from_directory(
                validation_dir, 
                target_size=(self.img_width, self.img_height), batch_size=self.batch_size,
                class_mode='categorical', interpolation="bilinear", seed=2021)

        opt_adam = keras.optimizers.Adam(lr=self.INIT_LR)
        # opt_sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

        model.compile(loss="categorical_crossentropy", optimizer=opt_adam, metrics=['accuracy'])

        log_dir = result_train_folder + '/' + 'log_'  +  self.model_name
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.checkpoint_dir = os.path.join(result_train_folder , "checkpoint_" + self.model_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        checkpoint_path = self.checkpoint_dir + "/cp_{epoch:02d}.h5"

        call_back = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True), 
                    tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path ),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, mode='auto', restore_best_weights=False)  
                    ]

        history = model.fit(train_generator, epochs=self.EPOCHS, validation_data=validation_generator, verbose=1, callbacks=call_back)

        end = datetime.datetime.now()
        delta = str(end-start)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # End statement
        with open(training_output_txt, 'w') as f:
            print("====================================================")
            print("\n Time taken (h/m/s): %s" %delta[:7], file=f)
            print("====================================================", file=f)
            print("\n Loss       " + ' '.join(str(round(e,4)) for e in loss) , file=f)
            print("\n Val. Loss  " + ' '.join(str(round(e,4)) for e in val_loss) , file=f)
            print("--------------------------------------------", file=f)
            print("\n Acc.       " + ' '.join(str(round(e,4)) for e in acc) , file=f)
            print("\n Val. Acc.  " + ' '.join(str(round(e,4)) for e in val_acc) , file=f)
            print("====================================================", file=f)


    def eval(self):
        if not os.path.exists(self.checkpoint_dir):
            print("No checkpoint at the path, check again!")
            return -1

        for cp_index in os.listdir(self.checkpoint_dir):
            checkpoint_path = os.path.join(self.checkpoint_dir, cp_index)
            model = load_model(checkpoint_path)
            input_model = model.input_shape
            width , height = input_model[1], input_model[2]
            print(width, height)

            result_test_folder = self.result_folder + '/test_'  + cp_index[:-3]
            if not os.path.isdir(result_test_folder):
                os.makedirs(result_test_folder)

            result_txt = result_test_folder + '/result_test.txt'
            spoof_score_txt = result_test_folder + '/score_prediction.txt'
            wrong_sample_path = result_test_folder + '/wrong_sample_path.txt'
        
            valid_datagen = ImageDataGenerator()   
            validation_dir = crop_data_test
            validation_generator = valid_datagen.flow_from_directory( validation_dir,
                    target_size=(width, height), batch_size=1, shuffle= False,
                    class_mode='categorical', interpolation="bilinear", seed=2021)
            filenames = validation_generator.filenames
            nb_samples = len(filenames)
            prediction = model.predict(validation_generator, steps = nb_samples)
            spoof_score = np.array(prediction)[:,1]
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
            test_len = len(prediction)
            wrong_list = []
            predict_live = 0
            wrong_spoof = 0
            for i in range(test_len):
                if prediction_class[i] == 0:
                    predict_live += 1
                    if labels[i] == 1 :
                        wrong_spoof += 1
                        wrong_list.append(i)
            if predict_live == 0:
                print('No prediction is live', file=open(result_txt, 'a'))
                wrong_rate = 0
            else:
                print(f"number of spoof samples is predicted as live is {wrong_spoof}", file=open(result_txt, 'a'))
                wrong_rate = round(wrong_spoof/count_live, 4)
            print(f"model predict number of sample as live : {predict_live}", file=open(result_txt, 'a'))
            print(f"model has wrong live rate (BPCER) = {wrong_rate} ", file=open(result_txt, 'a'))
            predict_spoof = 0
            wrong_live = 0
            for i in range(test_len):
                if prediction_class[i] == 1:
                    predict_spoof += 1
                    if labels[i] == 0 :
                        wrong_live += 1
                        wrong_list.append(i)
            if predict_spoof == 0:
                print('No prediction is spoof')
                wrong_rate = 0
            else:
                print(f"number of live samples is predicted as spoof is {wrong_live}", file=open(result_txt, 'a'))
                wrong_rate = round(wrong_live/count_spoof, 4)
            print(f"model predict number of sample as spoof : {predict_spoof}", file=open(result_txt, 'a'))
            print(f"model has wrong spoof rate (APCER) = {wrong_rate}", file=open(result_txt, 'a'))
            avg_wrong_rate = round((wrong_live + wrong_spoof)/(count_live + count_spoof), 4)
            print(f"the average wrong rate of model is: {avg_wrong_rate}", file=open(result_txt, 'a'))
            wrong_path = []
            for sample_index in wrong_list:
                wrong_path.append(filenames[sample_index])
            with open(wrong_sample_path, 'w') as f:
                for item in wrong_path:
                    f.write("%s\n" % item)


if __name__ == '__main__':

    # b0_ver_1 = config_to_train(model_name='b0_ver_1', build_model=build_b0_gap, INIT_LR=1e-4 )
    # b0_ver_1.train()
    # b0_ver_1.eval()

    b0_ver_2 = config_to_train(model_name='b0_ver_2', build_model=build_b0_gap, INIT_LR=3e-4 )
    b0_ver_2.train()
    b0_ver_2.eval()

    # b0_ver_3 = config_to_train(model_name='b0_ver_3', build_model=build_b0_fully_connected, INIT_LR=1e-4 )
    # b0_ver_3.train()
    # b0_ver_3.eval()