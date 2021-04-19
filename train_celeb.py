import numpy as np
import keras
import tensorflow as tf
from keras.applications import VGG16
import os, datetime
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
import statistics
# my packages
from config import *
from model_zoo import *
from eer_calculation import cal_metric

start = datetime.datetime.now()

train_datagen = ImageDataGenerator(
#       rotation_range=40,
#       width_shift_range=0.2,
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2,
#       horizontal_flip=True,
#       fill_mode='nearest'
)

# Note that the validation data should not be augmented!
valid_datagen = ImageDataGenerator()   

train_dir = crop_data_train
train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 224x224
        target_size=(224, 224),
        batch_size=batch_size,
        # Since we use categorical_crossentropy loss, we need binary labels
        class_mode='categorical')

validation_dir = crop_data_test
validation_generator = valid_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')

#with open('number_sample.txt', 'r') as f:
#    a = f.readlines()
#    number_train_sample = int(a[0])
#    number_valid_sample = int(a[1])

# compile model
# resnet 50
# model = build_resnet50(width=image_size, height=image_size, depth=image_depth, classes=2)
# efficent net b4
# model = build_efficient_b7(width=image_size, height=image_size, depth=image_depth, classes=2)

model = build_efficient_net_b4(224, 2)

model.compile(loss="categorical_crossentropy", optimizer=opt_sgd, metrics=["accuracy"])

log_dir = folder_save_log + '/' +  model_name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

checkpoint_dir = os.path.join(work_place , "training_checkpoint", model_name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = checkpoint_dir + "/cp_{epoch:02d}.hdf5"

call_back = [tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=True), 
             tf.keras.callbacks.ModelCheckpoint( filepath=checkpoint_path ),
             tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, mode='auto', restore_best_weights=False)   ]

history = model.fit(train_generator,
      epochs=EPOCHS, validation_data=validation_generator, 
      verbose=2,
      callbacks=call_back)

# save the network to disk
print("[INFO] serializing network to drive ... ", file=open('result_training_output.txt', 'w'))
model.save(folder_save_model + '/' + model_name, save_format="h5")
print("complete save model", file=open('result_training_output.txt', 'a'))

end = datetime.datetime.now()
delta = str(end-start)

acc = history.history['accuracy']
acc = acc[-5:]
val_acc = history.history['val_accuracy']
val_acc = val_acc[-5:]
loss = history.history['loss']
loss = loss[-5:]
val_loss = history.history['val_loss']
val_loss = val_loss[-5:]

# End statement
with open('result_training_output.txt', 'a') as f:
    print("============================================")
    print("\n Time taken (h/m/s): %s" %delta[:7], file=f)
    print("============================================")
    print("\n Loss       " + ' '.join(str(e) for e in loss) , file=f)
    print("\n Val. Loss  " + ' '.join(str(e) for e in val_loss) , file=f)
    print("--------------------------------------------")
    print("\n Acc.       " + ' '.join(str(e) for e in acc) , file=f)
    print("\n Val. Acc.  " + ' '.join(str(e) for e in val_acc) , file=f)
    print("============================================")


# keras evaluate on validation data
model.evaluate(validation_generator, batch_size=1)




