import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla
import math

from PIL import Image, ImageDraw

import synthetic as synth

DEBUG = False

radius = 3

def get_entry_point():
    return 'gaze_draw'

def fourdigits(i):
    s = str(i)
    while (len(s) < 4):
        s = "0" + s
    
    return s    

def inBounds(x, y):
    width = 256
    height = 144
    if (x < 0 or y < 0):
        return False
    if (x >= width or y >= height):
        return False
    
    return True
    
def draw_gaze(path, px, py, sp, show=False):
    im = Image.open(path).convert("RGBA")
    img = ImageDraw.Draw(im)
    shape = (px - radius, py - radius, px + radius, py + radius)
    if (inBounds(px, py)):
        img.ellipse(shape, fill ="#FFFFFF", outline ="red")
    im.save(sp)
    if show:
        cv2.imshow("gaze_im", np.array(im))
        cv2.waitKey(1)
    
    
def solve(frame, datapoint, gaze_path, state, actors, cams):
    pathtodata = gaze_path[:gaze_path.index("/actordists")]
    # print(frame, " = ", datapoint)
    if (DEBUG):
        print("!!!!!!!!!!!!!", state)
    
    
    
    tailend = len(state) - 1
    while (tailend >= 0):
        if (not state[tailend].isdigit()):
            break
        tailend -= 1
    
    number = state[tailend + 1:]
    if (state == "vanishing_point"):
        number = -1
    else:
        number = int(number)
    # print("ID", number)
    
    pixloc = [128, 72]
    pixloc1 = [-1000, -1000]
    pixloc2 = [-1000, -1000]
    
    if (number >= 0):
        index = -1
        for i in range(len(actors)):
            if (str(actors[i].id) == str(number)):
                index = i
                break
        
        
        if (DEBUG):
            print(actors[index], number, "!!!!")
        
        pixloc = synth.obtainPixel(actors[index].get_transform(), cams[0])
        pixloc1 = synth.obtainPixel(actors[index].get_transform(), cams[1])
        pixloc2 = synth.obtainPixel(actors[index].get_transform(), cams[2])
        if (DEBUG):
            print(pixloc1, pixloc, pixloc2)
    else:
        if (DEBUG):
            print("VANISHING POINT!!!!")
    
    draw_gaze(pathtodata + "/rgb/" + fourdigits(frame) + ".png", pixloc[0], pixloc[1], pathtodata + "/rgbgaze/" + fourdigits(frame) + ".png", show=True)
    draw_gaze(pathtodata + "/rgb_left/" + fourdigits(frame) + ".png", pixloc1[0], pixloc1[1], pathtodata + "/rgbleftgaze/" + fourdigits(frame) + ".png")
    draw_gaze(pathtodata + "/rgb_right/" + fourdigits(frame) + ".png", pixloc2[0], pixloc2[1], pathtodata + "/rgbrightgaze/" + fourdigits(frame) + ".png")
