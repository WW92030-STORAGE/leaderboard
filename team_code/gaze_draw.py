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

import attentionMapCreator as mapc # REPLACE WITH THE CORRECT PATH

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
    
def draw_gaze(path, px, py, sp):
    '''
    im = Image.open(path).convert("RGBA")
    img = ImageDraw.Draw(im)
    shape = (px - radius, py - radius, px + radius, py + radius)
    if (inBounds(px, py)):
        img.ellipse(shape, fill ="#FFFFFF", outline ="red")
    
    im.save(sp)
    '''
    
    im = mapc.draw_attention_map([[int(px), int(py)]], Image.open(path))
    
    im[0].save(sp)

def tup(x):
    return int(x[0]), int(x[1])
    
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
    
    draw_gaze(pathtodata + "/rgb/" + fourdigits(frame) + ".png", pixloc[0], pixloc[1], pathtodata + "/rgbgaze/" + fourdigits(frame) + ".png")
    draw_gaze(pathtodata + "/rgb_left/" + fourdigits(frame) + ".png", pixloc1[0], pixloc1[1], pathtodata + "/rgbleftgaze/" + fourdigits(frame) + ".png")
    draw_gaze(pathtodata + "/rgb_right/" + fourdigits(frame) + ".png", pixloc2[0], pixloc2[1], pathtodata + "/rgbrightgaze/" + fourdigits(frame) + ".png")
    
    place = str(pathtodata) + "/points.txt"
            
    if (frame == 0):
        savefile = open(place, "x")
    else:
        savefile = open(place, "a")
    
    savefile.write(str(frame) + " = " + str(pixloc1[0]) + " " + str(pixloc1[1]) + " / " + str(pixloc[0]) + " " + str(pixloc[1]) + " / " + str(pixloc2[0]) + " " + str(pixloc2[1]))
    savefile.write("\n")
    
    savefile.close()
    
    
    # NumPy things
    
    """
    
    binarray = np.zeros((144, 256))
    binarray1 = np.zeros((144, 256))
    binarray2 = np.zeros((144, 256))
    if (inBounds(pixloc[0], pixloc[1])):
        binarray[int(pixloc[1])][int(pixloc[0])] = 1
    if (inBounds(pixloc1[0], pixloc1[1])):
        binarray1[int(pixloc1[1])][int(pixloc1[0])] = 1
    if (inBounds(pixloc2[0], pixloc2[1])):
        binarray2[int(pixloc2[1])][int(pixloc2[0])] = 1
        
    """
    
    binarray = mapc.get_attention_map([tup(pixloc)], (144, 256))
    binarray1 = mapc.get_attention_map([tup(pixloc1)], (144, 256))
    binarray2 = mapc.get_attention_map([tup(pixloc2)], (144, 256))
    
    np.save(pathtodata + "/matrix/" + fourdigits(frame) + ".npy", binarray)
    np.save(pathtodata + "/matrixleft/" + fourdigits(frame) + ".npy", binarray1)
    np.save(pathtodata + "/matrixright/" + fourdigits(frame) + ".npy", binarray2)
    
    
    
    
    
