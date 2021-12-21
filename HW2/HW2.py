# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 23:36:59 2021

@author: leo90
"""

import os
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('ignore')

x_ray_df = pd.read_csv('Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('D:\AI_Medical_dataset\Chest_Xrays','images*', '*', '*.png'))}
# print('Scans found:', len(all_image_paths), ', Total Headers', x_ray_df.shape[0])
x_ray_df['path'] = x_ray_df['Image Index'].map(all_image_paths.get)
# print(x_ray_df.sample(3))

x_ray_df['Finding Labels']=x_ray_df['Finding Labels'].map(lambda x: 'Unhealthy' if x != "No Finding" else "Healthy" )
print(x_ray_df['Finding Labels'].value_counts())

x_ray_df['Output'] = x_ray_df['Finding Labels'].map({'Unhealthy': 1 , 'Healthy': 0 })
# print(x_ray_df['Finding Labels'])

List=['Image Index','path', 'Finding Labels','Output']
x_ray_df=pd.DataFrame(data=x_ray_df,columns=x_ray_df[List].columns)
a=(x_ray_df[x_ray_df['Finding Labels'].isin(['Healthy'])][0:2500]).index.values
x_ray_df=x_ray_df.drop(a)
# print(x_ray_df['Finding Labels'].value_counts())

train_df, valid_df = train_test_split(x_ray_df,test_size = 0.15,random_state = 100)
# print('train', train_df.shape[0], 'Validation', valid_df.shape[0])
training_samples=train_df.shape[0]
validation_samples = valid_df.shape[0]

# train_labels=train_df['Finding Labels'].values
# validation_labels = valid_df['Finding Labels'].values
# Label_Encoder = LabelEncoder()
# Label_Encoder.fit(train_labels)
# # encode wine type labels
# train_labels_enc = Label_Encoder.transform(train_labels)
# validation_labels_enc = Label_Encoder.transform(validation_labels)

# # print(train_labels[0:5], train_labels_enc[0:5])
#IMPORTANT parameter

IMG_SIZE = (150,150)
NUM_CLASSES = 2
EPOCHS = 10
INPUT_SHAPE = (150,150, 3)


idg_augment = ImageDataGenerator(rescale=1./255,samplewise_center=True,samplewise_std_normalization=True, 
                              horizontal_flip = True,vertical_flip = False,height_shift_range= 0.05, 
                              width_shift_range=0.1,  rotation_range=5,shear_range = 0.1,
                              fill_mode = 'reflect', zoom_range=0.15)

def flow_from_dataframe(img_data_gen, in_df, **dflow_args):
    base_dir = os.path.dirname(in_df['path'].values[0])
    df_gen = img_data_gen.flow_from_directory(base_dir, 
                                     class_mode = 'binary',
                                    **dflow_args)
    df_gen.filenames = in_df['path'].values
    df_gen.classes = np.stack(in_df['Output'].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = '' # since we have the full path
    df_gen.filepaths.extend(df_gen.filenames)
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


train_gen = flow_from_dataframe(idg_augment, train_df,color_mode='rgb',
                                batch_size = 100,target_size = IMG_SIZE)

valid_gen = flow_from_dataframe(idg_augment, valid_df,color_mode='rgb', 
                                 batch_size = 150,target_size = IMG_SIZE)

test_gen= flow_from_dataframe(idg_augment,valid_df,color_mode='rgb',
                              target_size = IMG_SIZE,batch_size = 1)

steps_per_epoch= (training_samples )/100
val_steps = validation_samples/150
#transfer learning
#Build Model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
#Freeze Layers
for layer in vgg_model.layers[:-9]:
    layer.trainable = False

for layer in vgg_model.layers:
    print(layer, layer.trainable)

model = Sequential()
model.add(vgg_model)
model.add(Dense(512, activation='relu', input_dim=INPUT_SHAPE ))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=optimizers.Adam(lr=1e-5),
              metrics=['accuracy'])

model.summary()


history = model.fit(train_gen,validation_data = valid_gen,epochs = EPOCHS,
                    steps_per_epoch= steps_per_epoch,validation_steps= val_steps)
model.save('VGG16_NIHxRays_model.h5')     #將模型儲存至 HDF5檔案中
print("VGG16_NIHxRays_model.h5 模型儲存完畢!")


print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
















