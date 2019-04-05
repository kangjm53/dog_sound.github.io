#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:29:06 2019

@author: kangjm
"""

#from tkinter import *
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sys
import soundfile as sf
import sounddevice as sd
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.ndimage.interpolation import zoom


class app():    
    def __init__(self):
        defailt_fs = 44100
        sd.default.device
        sd.default.samplerate = defailt_fs
        
        json_file = open("model_dogsound.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        self.loaded_model.load_weights("model_dogsound.h5")
        print("Loaded model from disk")
        self.loaded_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            
        self.root=tk.Tk()
        self.root.title("Dog Sound")
        self.root.geometry("200x100")
        self.root.resizable(0,0)
        # Code to add widgets will go here...
        lbl=tk.Label(self.root,text='File search')
        lbl.grid(row=0,column=0)
        
        b_f_o=tk.Button(self.root,text='File open',command=self.file_open)
        b_f_o.grid(row=0,column=1)
        
        
        b_rec=tk.Button(self.root,text='Rec',command=self.rec,)
        b_rec.grid(row=3,column=0)
        
        b_stop=tk.Button(self.root,text='Stop',command=self.stop)
        b_stop.grid(row=3,column=1)
        
        b_play=tk.Button(self.root,text='Play',command=self.play)
        b_play.grid(row=4,column=1)
        
        b_exec=tk.Button(self.root,text='Excute',command=self.exe)
        b_exec.grid(row=4,column=2)
        
        exit_btn=tk.Button(self.root,text='Exit',command=self.qut)
        exit_btn.grid(row=5,column=0)
    

    def file_open(self) :
        #fo=tk.Tk()
        self.root.filename = filedialog.askopenfilename(initialdir='.',title='Select file',filetypes = (("Wave files","*.wav"),("all files","*.*")))
        self.root.update()
        
    def wav2img(self,data,fs):
    
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
        
        
    def rec(self) :
        duration = 5.5
        self.myrecording = self.sd.rec(int(duration * default_fs), samplerate=default_fs, channels = 1)
        print('rec')
    
    def stop(self) :
        self.sd.wait()
        print('stop')

    def play(self) :
        self.sd.play(self.myrecording, default_fs)
        print('stop')
    
    def exe(self) :
        print(self.root.filename)
        if self.root.filename == None :
            print("Please select file or record sound.")

        else :
            # model evaluation
            data, fs = sf.read(self.root.filename, dtype='float32')
            print(self.root.filename,data.shape)
            X=self.wav2img(data[:,0],fs)
            X=X.reshape(1,70,70,1)
            #print(self.loaded_model)
            score = self.loaded_model.predict(X).reshape((-1))
            print(type(score))
            print("%s : " % self.loaded_model.metrics_names[1])
            print("%.2f%% %.2f%% %.2f%% %.2f%% %.2f%% %.2f%%" % (score[0],score[1],score[2],score[3],score[4],score[5]))
            
    def qut(self):
        self.root.withdraw()
        self.root.update()
        #sys.exit()
        self.root.destroy()
        
if __name__== "__main__" :
    window = app()
    window.root.mainloop()
    
    print(window.root.filename)
    
    

