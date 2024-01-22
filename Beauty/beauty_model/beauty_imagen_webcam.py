
"""
This is a kind of dummy script that mimics one state-action pair of the Beauty robot's actions.
It mimics commands from the rl model's controller to select a pipette, collect attract/repellent solution
and drop it on the plate at a particular location (that the controller determines) before returning to
its home position.
"""

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Input, Dense, Dropout, Lambda, Reshape, MaxPooling2D, LSTM, Reshape
# from tensorflow.keras.models importkdl,kkk'sdee
import os
import sys
import serial
import threading
# import cv2
import time
from time import sleep
import numpy as np
import gphoto2 as gp
import argparse
import random
import math

print('testing interpreter and script - run is good..')

import cv2

counter = 0
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

load_txt = "Loading RNN / World Model..."
action_txt = "Model loaded. Showing image to World Model."
image_txt = ""
coord_txt = "Next action: Peptone Solution, 140.5, -129.41, -5"


print(load_txt)
font = cv2.FONT_HERSHEY_SIMPLEX
org = (400,300)
org2 = (400,300+50)
org3 = (400,300+100)
org4 = (400,300+150)
org5 = (400,300+200)
font_scale = 1
color = (80,0,0)
thickness = 1

while True:
    ret, frame = cap.read()
    dsize = (1920, 1080)
    # frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    frame = cv2.resize(frame, dsize, interpolation=cv2.INTER_AREA)
    frame = cv2.putText(frame, load_txt, org, font, font_scale, color, thickness, cv2.LINE_AA)
    if counter > 150:
        frame = cv2.putText(frame, action_txt, org2, font, font_scale, color, thickness, cv2.LINE_AA)
        
    if counter > 200:
        frame = cv2.putText(frame, coord_txt, org3, font, font_scale, color, thickness, cv2.LINE_AA)
        
    if counter > 350:
        image_txt = "Action completed. Showing next image to World Model."
        frame = cv2.putText(frame, image_txt, org4, font, font_scale, color, thickness, cv2.LINE_AA)
        
    
    if counter > 400:
        next_action_txt = "Next action: Dextrose Solution, 140.5, -177.41, -5"
        frame = cv2.putText(frame, next_action_txt, org5, font, font_scale, color, thickness, cv2.LINE_AA)
        
    if counter > 2750:
        image_txt = "Action completed. Showing next image to World Model."
        frame = cv2.putText(frame, image_txt, (400,300+250), font, font_scale, color, thickness, cv2.LINE_AA)
        
    
    if counter > 2850:
        next_action_txt = "Next action: Chloramphenicol Solution, 172.5, -161.41, -5"
        frame = cv2.putText(frame, next_action_txt, (400,300+300), font, font_scale, color, thickness, cv2.LINE_AA)
        
    
    if counter > 3050:
        image_txt = "Action completed. Showing next image to World Model."
        frame = cv2.putText(frame, image_txt, (400,300+350), font, font_scale, color, thickness, cv2.LINE_AA)
        
    
    if counter > 3150:
        next_action_txt = "Next action: Peptone Solution, 140.5, -145.41, -5"
        frame = cv2.putText(frame, next_action_txt, (400,300+400), font, font_scale, color, thickness, cv2.LINE_AA)
        
    if counter > 4400:
        counter = 0
    
    cv2.imshow('Input', frame)
    
    
    counter += 1
    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()


