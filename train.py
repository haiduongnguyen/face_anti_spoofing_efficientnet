import numpy as np
import keras
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_lattice as tfl
import os, datetime
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import statistics
# my packages
from config import *
from model_zoo import *
from eer_calculation import cal_metric
# from focal_loss import BinaryFocalLoss, SparseCategoricalFocalLoss
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

start = datetime.datetime.now()
from focal_losses import categorical_focal_loss



# image parameter
# b0 = 224, b1 = 240, b4 = 380
image_width = 224
image_height = 224
image_depth = 3

INIT_LR = 1e-4
batch_size = 8
EPOCHS = 20
patience = 5

# model_name = 'efficient_net_b1_ver03'
# model = build_efficient_net_b1(image_size, image_depth, 2)

# model_name = 'efficient_net_b4_ver06'
# model = build_efficient_net_b4(image_size, image_depth, 2)

# model_name = 'new_b4_ver01'
# image_size = 380
# model = build_new_efficient_net_b4(380, 380, 3, 2)

# model_name = 'densenet121_ver01'
# model = build_dense_net121(image_size, image_size, image_depth, 2)

# model_name = 'lam_resnet_ver02'
# model = build_lamresnet50(image_size, image_size, image_depth, 2)


# model_name = 'new_efficient_b0_ver01'
# model = build_new_efficient_net_b0(image_size, image_size, image_depth, 2)

# model_name = 'new_b0_ver0'
# model = build_new_efficient_net_b0(image_size, image_size, image_depth, 2)

# model_name = 'new_b0_add_convolutional_layer'
# model = build_new_b0_add_convolutional_layer(224,224,3,2)

# model_name = 'new_b0_ver4'
# model = build_new_efficient_net_b0(image_size, image_size, image_depth, 2)


# model_name = 'new_b0_ver5'
# model = build_new_efficient_net_b0(image_width, image_height, image_depth, 2)


# model_name = 'new_b0_ver6'
# model = build_new_efficient_net_b0(image_width, image_height, image_depth, 2)

model_name = 'new_b0_ver7'
model = build_new_efficient_net_b0(image_width, image_height, image_depth, 2)

# model_name = 'new_b1_ver1'
# model = build_new_efficient_net_b1(image_width,image_height, image_depth, 2)

# model_name = 'new_b1_ver2'
# model = build_new_efficient_net_b1(image_width,image_height, image_depth, 2)

# model_name = 'new_b4_ver2'
# model = build_new_efficient_net_b4(image_width, image_height, image_depth, 2)


result_folder = work_place + '/result_' + model_name
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)
result_train_folder = result_folder + '/train'
if not os.path.isdir(result_train_folder):
    os.makedirs(result_train_folder)

training_output_txt = result_train_folder + '/result_training_output.txt'


train_datagen = ImageDataGenerator(
      rotation_range=45,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest',
      brightness_range=[0.5,1.5]
)

train_dir = crop_data_train
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 224x224
        target_size=(image_width, image_height),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical',
        # interpolation="bilinear", 
        interpolation="bilinear", 
        seed=2021)

# Note that the validation data should not be augmented!
valid_datagen = ImageDataGenerator()   
validation_dir = crop_data_test
validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_width, image_height),
        batch_size=batch_size,
        class_mode='categorical',
        interpolation="bilinear",
        seed=2021)

focal_loss = tfa.losses.SigmoidFocalCrossEntropy()

opt_adam = keras.optimizers.Adam(lr=INIT_LR)
opt_sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)

model.compile(loss="categorical_crossentropy", optimizer=opt_adam, metrics=['accuracy'])

log_dir = result_train_folder + '/' + 'log_'  +  model_name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

checkpoint_dir = os.path.join(result_train_folder , "checkpoint")
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = checkpoint_dir + "/cp_{epoch:02d}.h5"

call_back = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True), 
             tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path ),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, mode='auto', restore_best_weights=False)  
            ]

history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator, verbose=1, callbacks=call_back)

# model.save(result_train_folder + '/' + model_name + '.h5')

end = datetime.datetime.now()
delta = str(end-start)

acc = history.history['accuracy']
acc = acc
val_acc = history.history['val_accuracy']
val_acc = val_acc
loss = history.history['loss']
loss = loss
val_loss = history.history['val_loss']
val_loss = val_loss

# End statement
with open(training_output_txt, 'w') as f:
    print("============================================")
    print("\n Time taken (h/m/s): %s" %delta[:7], file=f)
    print("============================================")
    print("\n Loss       " + ' '.join(str(e) for e in loss) , file=f)
    print("\n Val. Loss  " + ' '.join(str(e) for e in val_loss) , file=f)
    print("--------------------------------------------")
    print("\n Acc.       " + ' '.join(str(e) for e in acc) , file=f)
    print("\n Val. Acc.  " + ' '.join(str(e) for e in val_acc) , file=f)
    print("============================================")