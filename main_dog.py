#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 14:29:06 2019

@author: kangjm
"""

#from tkinter import *
import os
import tkinter as ttk
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
        json_file = open("model_dogsound.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = tf.keras.models.model_from_json(loaded_model_json)
        self.loaded_model.load_weights("model_dogsound.h5")
        print("Loaded model from disk")
        self.loaded_model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            
        self.dist=[]
        with open('list.txt','r') as f :
            self.dist=f.read()
            self.dist=self.dist.split()
        #print(self.dist)
        
        self.root=ttk.Tk()
        self.root.title("GaeGongGam")
        self.root.geometry("200x450+100+100")
        self.root.resizable(1,1)
        #filepath='/home/pi/Documents/project/'
        
        self.frame1=ttk.Frame(self.root, relief="solid", bd=2)
        self.frame1.pack(side="top", fill="both", expand=True)
        
        self.frame2=ttk.Frame(self.root, relief="solid", bd=2)
        self.frame2.pack(side="bottom", fill="both", expand=True)
        
        filepath='/Users/kangjm/Documents/project/'
        self.img=ttk.PhotoImage(file=filepath+'128px-Barking_Dog.png')
        self.label1=ttk.Label(self.frame1, image=self.img)
        self.label1.pack() 

        b_f_o=ttk.Button(self.frame1,text='OPEN : Dog Sound File',command=self.file_open)
        b_f_o.pack()
        
        
        b_rec=ttk.Button(self.frame1,text='Record : Dog Sound Record',command=self.rec)
        b_rec.pack()
       
        lbl1=ttk.Label(self.frame1,text='-------------------')
        lbl1.pack()
        
        b_play=ttk.Button(self.frame1,text='Sound : P l a y',command=self.play)
        b_play.pack()
        
        b_stop=ttk.Button(self.frame1,text='Sound : S t o p',command=self.stop)
        b_stop.pack()

        b_exec=ttk.Button(self.frame1,text='What my doogy wants?',command=self.exe)
        b_exec.pack()
        
        lbl2=ttk.Label(self.frame1,text='-------------------')
        lbl2.pack()
        
        self.exit_btn=ttk.Button(self.frame1,text='Exit',command=self.qut)
        self.exit_btn.pack()
        
        self.img2=ttk.PhotoImage(file=filepath+'128px-Barking_Dog.png')
        self.label2=ttk.Label(self.frame2, image=self.img2)
        self.label2.pack()        

    def file_open(self) :
        self.root.filename = filedialog.askopenfilename(initialdir='.',title='Select file',filetypes = (("Wave files","*.wav"),("all files","*.*")))
        self.root.update()
        data, fs = sf.read(self.root.filename, dtype='float32')
        self.data=data[:,0]
        self.fs=fs
        self.flag=1
        #print(self.root.filename,data.shape)
        
    def wav2img(self,data,fs):
    
        N=data.shape[0]
        T=1./fs
        if N*T > 5 :
            data=data[:fs*10]
        N=data.shape[0]
        f_data=fft(data)
        yf = fft(data)
        xf = fftfreq(N, T)
        xf = fftshift(xf)
        f_plot = fftshift(yf)
        xf=xf[int(N/2)-1:]
        f_plot=f_plot[int(N/2)-1:]
        
        N=f_plot.shape[0]
     
        x_r=zoom(xf,4900./N)
        f_r=zoom(1.0/N * np.abs(f_plot),4900./N)
      
        image_f=x_r.reshape(70,70)
        print(type(f_r),f_r.shape,f_r)
        image=f_r.reshape(70,70)
        image=image/image.max()

        return image

    def rec(self) :
        duration = 5
        self.rfs=48000
        self.myrec = sd.rec(int(duration * self.rfs), samplerate=self.rfs, channels = 1)
        self.myrec=self.myrec.reshape(-1,)
        self.flag=2
        
    def stop(self) :
        sd.stop()

    def play(self) :
        if self.flag==1 :sd.play(self.data, self.fs)
        if self.flag==2 :sd.play(self.myrec, self.rfs)
    
    def exe(self) :
        if self.flag ==1 :
            X=self.wav2img(self.data,self.fs)
            X=X.reshape(1,70,70,1)

            score = self.loaded_model.predict(X).reshape((-1))
            print(type(score))
            print("%s : " % self.loaded_model.metrics_names[1])
            for i in range(len(self.dist)):
                print("%s : %.2f%%" % (self.dist[i],score[i]*100.))
            feeling=max(range(len(score)), key = lambda x: score[x])
            print(feeling)
            self.popup_img(feeling)
            
        elif self.flag==2 :
            print(type(self.myrec))
            X=self.wav2img(self.myrec,self.rfs)
            X=X.reshape(1,70,70,1)

            score = self.loaded_model.predict(X).reshape((-1))
            print(type(score))
            print("%s : " % self.loaded_model.metrics_names[1])
            for i in range(len(self.dist)):
                print("%s : %.2f%%" % (self.dist[i],score[i]*100.))
            feeling=max(range(len(score)), key = lambda x: score[x])
            print(feeling)
            self.popup_img(feeling)
        else :
            print("Please select file or record sound.")

            
    def qut(self):
        self.root.withdraw()
        self.root.update()
        sys.exit()
        
            
    def popup_img(self,feeling):
        self.img2=ttk.PhotoImage(file='./feeling_img/'+self.dist[feeling]+'.png')
        self.label2.configure(image=self.img2)
        self.label2.image=self.img2
        #self.root.update()
        #label2=ttk.Label(self.frame2, image=self.img2)
        #label2.pack()
        

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def main():
    window = app()
    window.root.mainloop()
    
    print(window.root.filename)
      
if __name__== "__main__" :
    main()    
    

