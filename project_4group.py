#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 13:33:10 2019

@author: kangjm
"""
import os
#import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import tensorflow as tf

#from keras.models import Sequential
#from keras.layers import Dense
#from keras.callbacks import ModelCheckpoint
#import pandas as pd

from scipy.fftpack import fft, fftfreq, fftshift
from scipy.ndimage.interpolation import zoom
from sklearn.model_selection import train_test_split

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def wav2img(data,fs):
    
    N=data.shape[0]
    T=1./fs
    if N*T > 10 :
        data=data[:fs*10]
    N=data.shape[0]
    f_data=fft(data)
    yf = fft(data)  
    xf = fftfreq(N, T)
    xf = fftshift(xf)
    f_plot = fftshift(yf)
    xf=xf[int(N/2)-1:]
    f_plot=f_plot[int(N/2)-1:]
    #plt.xlim((20,20000))
    #plt.ylim((0,0))
    N=f_plot.shape[0]
    #plt.plot(xf, 1.0/N * np.abs(f_plot))
    
    x_r=zoom(xf,4900./N)
    f_r=zoom(1.0/N * np.abs(f_plot),4900./N)
    #plt.plot(x_r, f_r)
    
    image_f=x_r.reshape(70,70)
    image=f_r.reshape(70,70)
    image=image/image.max()
#    plt.figure()
#    plt.imshow(image)
#    plt.colorbar()
#    plt.grid(False)
#    plt.show()
    return image
    
#filepath='/home/pi/Documents/project/Dog_sound/'
filepath='/Users/kangjm/Documents/project/dog_sound/'
folder_list=list(listdir_nohidden(filepath))
print(folder_list)

n_folder=len(folder_list)
with open('list.txt','w') as f :
    for i in folder_list:
        d_write="%s " % i
        f.write(d_write)


n_file=0
for i in range(n_folder):
    file_list=list(listdir_nohidden(filepath+folder_list[i]))
    n_file+=len(file_list)

test_X=np.zeros((n_file,70,70))
test_y=np.zeros((n_file))

k=0
for i in range(n_folder):
    file_list=list(listdir_nohidden(filepath+folder_list[i]))
    n_file=len(file_list)
    for j in range(n_file):
        data, fs = sf.read(filepath+folder_list[i]+'/'+file_list[j], dtype='float32')
        #print(folder_list[i]+'/'+file_list[j],data.shape)
        data_angry=wav2img(data[:,0],fs)
        test_y[k]=i
        test_X[k]=data_angry
        k+=1
        
x_train, x_test, y_train, y_test = train_test_split(test_X, test_y, test_size=0.01)
    
x_train = x_train.reshape(x_train.shape[0],70,70,1)
x_test = x_test.reshape(x_test.shape[0],70,70,1)

model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(5,5), strides=(1,1),\
                               padding='same', activation='relu', input_shape=(70,70,1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.Conv2D(64, kernel_size=(5,5), strides=(2,2),\
                               padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024,activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_folder,activation='softmax')

])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train,epochs=10)
print('====')
model_ev=model.evaluate(x_test,y_test)

x_pred=x_train[0:3]
y_pred=y_train[0:3]

pred=model.predict(x_pred)
print(pred)

model_json = model.to_json()
with open("model_dogsound.json", "w") as json_file : 
    json_file.write(model_json)

model.save_weights("model_dogsound.h5")
print("Saved model to disk")
model.save("model_dogsound.ww")


