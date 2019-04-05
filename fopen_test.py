#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 15:02:14 2019

@author: kangjm
"""
from tkinter import filedialog
from tkinter import *
 
root = Tk()
root.withdraw()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("all files","*.*")))
print (root.filename)
