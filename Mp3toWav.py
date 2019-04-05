#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 15:35:58 2019

@author: kangjm
"""

from pydub import AudioSegment
filepath='/Users/kangjm/Documents/UST/행정/신입생예비교육/2019_1st/project/Dog_sound'
sound = AudioSegment.from_mp3(filepath+'/sad/sad1.mp3')
sound.export(filepath+'/sad/sad1.wav", format="wav")
