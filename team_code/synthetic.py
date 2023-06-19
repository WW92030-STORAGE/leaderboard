import os
import time
import datetime
import pathlib

import numpy as np
import cv2
import carla
import timeit

from PIL import Image, ImageDraw

from carla_project.src.common import CONVERTER, COLOR
from team_code.map_agent import MapAgent
from team_code.pid_controller import PIDController
import gaze_graph as gaga
import gaze_draw as gada
from gaze_graph import GazeTracker



tracker = GazeTracker(0.99, 0.01, 0.8)

def get_entry_point():
    return 'synthetic'
    
    
def dist(x1, x2, x3, y1, y2, y3):
    res = (y1 - x1) ** 2 + (y2 - x2) ** 2 + (y3 - x3) ** 2
    return np.sqrt(res)
        
def locdist(loc1, loc2):
    return dist(loc1.x, loc1.y, loc1.z, loc2.x, loc2.y, loc2.z)

def dot(v1, v2):
    return (v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

def rsq(v):
    return dot(v, v)

def norm(v):
    return np.sqrt(rsq(v))
    
def cross(v1, v2):
    return v1.y * v2.z - v1.z * v2.y + v1.z * v2.x - v1.x * v2.z + v1.x * v2.y - v1.y * v2.x

def vcos(v1, v2):
    return dot(v1, v2) / np.sqrt(rsq(v1) * rsq(v2))
    
def xy_dot(v1, v2):
    return (v1.x * v2.x) + (v1.y * v2.y)

def xy_rsq(v):
    return xy_dot(v, v)

def xy_norm(v):
    return np.sqrt(xy_rsq(v))

def xy_cos(v1, v2):
    return xy_dot(v1, v2) / np.sqrt(xy_rsq(v1) * xy_rsq(v2))

def proj(v1, v2): # project v1 onto the line v2
    return v2 * (dot(v1, v2) / dot(v2, v2))

def mat_trans(matrix, ref, off):
    location = carla.Location(matrix[0][3], matrix[1][3], matrix[2][3])
    rotation = carla.Rotation(ref.pitch + off.pitch, ref.yaw + off.yaw, ref.roll + off.roll)
    return carla.Transform(location, rotation)

def compdiv(v1, v2):
    return carla.Vector3D(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z)
    
def binstring(x, lenx):
    res = format(x, 'b')
    while (len(res) < lenx):
        res = "0" + res
    
    return res

def sgn(x):
    if (x > 0):
        return 1
    elif (x < 0):
        return -1
    return 0

def obtainPixel(actor, cam): # both are transforms
    disp = (actor.location).__sub__(cam.location)
    angle = np.arccos(xy_cos(disp, cam.get_forward_vector()))
    
    # print("RELATIVE POS", cam.get_forward_vector(), disp, angle)
        
    width = 256
    height = 144
    
    res = [0, 0]
    
    direct = proj(disp, cam.get_forward_vector())
    # print("CAM COMPONENT", direct, vcos(direct, cam.get_forward_vector()))
    
    therestofit = disp.__sub__(direct)
    # print("ORTHO", therestofit, vcos(therestofit, cam.get_forward_vector()))
    
    relativex = proj(therestofit, cam.get_right_vector())
    relativey = therestofit.__sub__(relativex)
    
    # print("COMPONENTS!!!!", relativex, relativey, relativex.__add__(relativey.__add__(direct)))
    # print("PERP CHECK!!!!", vcos(relativex, relativey), vcos(relativey, direct), vcos(direct, relativex))
    # print("DIR CHECK!!!!!", vcos(relativex, cam.get_right_vector()), vcos(relativey, cam.get_up_vector()), vcos(direct, cam.get_forward_vector()))
    # print("THINGS!!!!", compdiv(relativex, cam.get_right_vector()), compdiv(relativey, cam.get_up_vector()))
    
    res[0] = (width - width * norm(relativex) / norm(direct) * sgn(relativex.x)) * 0.5
    res[1] = (height - height * norm(relativey) / norm(direct) * sgn(relativey.y)) * 0.5
        
    return res


def collectData(step, world, vehicle, save_path, tick_data):
    TIME_BEGIN = timeit.default_timer()
    
    frame = step // 10
    
    # SYNTHETIC GAZE SYSTEMS
    Image.fromarray(tick_data['rgb']).save(save_path / 'rgb' / ('%04d.png' % frame))
    Image.fromarray(tick_data['rgb_left']).save(save_path / 'rgb_left' / ('%04d.png' % frame))
    Image.fromarray(tick_data['rgb_right']).save(save_path / 'rgb_right' / ('%04d.png' % frame))
    Image.fromarray(tick_data['topdown']).save(save_path / 'topdown' / ('%04d.png' % frame))
    
    actors = world.get_actors()
    
    # 0 = CENTER 1 = LEFT 2 = RIGHT
    
    cam0_loc = carla.Location(x=1.3, y=0, z=1.3)    
    cam0_rot = carla.Rotation(pitch=0, yaw=0, roll=0)
    
    cam1_loc = carla.Location(x=1.2, y=-0.25, z=1.3)
    cam1_rot = carla.Rotation(pitch=0, yaw=-45, roll=0)
    
    cam2_loc = carla.Location(x=1.2, y=0.25, z=1.3)
    cam2_rot = carla.Rotation(pitch=0, yaw=45, roll=0)
    
    cam0_trans = carla.Transform(location=cam0_loc, rotation=cam0_rot)
    cam1_trans = carla.Transform(location=cam1_loc, rotation=cam1_rot)
    cam2_trans = carla.Transform(location=cam2_loc, rotation=cam2_rot)
    
    vehicle_matrix = vehicle.get_transform().get_matrix()
    vehicle_transform = vehicle.get_transform() # VEHICLE TRANSFORM
    vehicle_forward = vehicle_transform.get_forward_vector() # VEHICLE FORWARD VECTOR
    
    # To get camera location you must do the following:
    # Rotate the offset by the vehicle forward vector
    # Translate the offset by the vehicle position
    
    cam0_absolute = np.matmul(vehicle.get_transform().get_matrix(), cam0_trans.get_matrix())    
    cam1_absolute = np.matmul(vehicle.get_transform().get_matrix(), cam1_trans.get_matrix()) 
    cam2_absolute = np.matmul(vehicle.get_transform().get_matrix(), cam2_trans.get_matrix()) 
    
    cam0 = mat_trans(cam0_absolute, vehicle_transform.rotation, cam0_rot)
    cam1 = mat_trans(cam1_absolute, vehicle_transform.rotation, cam1_rot)
    cam2 = mat_trans(cam2_absolute, vehicle_transform.rotation, cam2_rot)
    
    cams = [cam0, cam1, cam2]
    
    SYNTHETIC_GAZE = True
    DEBUG = False
        
    resultantActors = []
    ids = []
    types = []
    distances = []
    angles = []
    cam0x = []
    cam0y = []
    
    threshold = 50
    framesperslide = 65536
    
    if (step % 10 == 0 and SYNTHETIC_GAZE):
        data = dict() # data points for actors
        
        print("DATA FOR FRAME", frame)
        for index in range(max(0, len(actors))): # THIS APPLIES AN ACTION ON ALL WORLD ACTORS
            actor = actors[index]
                
            disp0 = actor.get_location().__sub__(cam0.location)
            angle0 = np.arccos(xy_cos(disp0, cam0.get_forward_vector()))
            disp1 = actor.get_location().__sub__(cam1.location)
            angle1 = np.arccos(xy_cos(disp1, cam1.get_forward_vector()))
            disp2 = actor.get_location().__sub__(cam2.location)
            angle2 = np.arccos(xy_cos(disp2, cam2.get_forward_vector()))
                
            isVisible = 7 # nonzero means visible
                
            if (angle0 > np.pi / 4 or xy_norm(disp0) > threshold):
                    isVisible = isVisible - 1
            if (angle1 > np.pi / 4 or xy_norm(disp1) > threshold):
                    isVisible = isVisible - 2
            if (angle2 > np.pi / 4 or xy_norm(disp2) > threshold):
                    isVisible = isVisible - 4
            
            
            avgdist = (xy_norm(disp0) + xy_norm(disp1) + xy_norm(disp2)) / 3.0
            
            if (actor.type_id == "traffic.unknown"):
                continue
            if (actor.type_id == "sensor.camera.rgb"):
                continue
            
            if (isVisible == 0):
                continue
                
            if (DEBUG):
                print(index, actor, actor.get_location(), " = ", avgdist)
                print(index, disp0, disp1, disp2)
                print(index, angle0, angle1, angle2, binstring(isVisible, 3))
                print("DOT WITH CAM 0", xy_dot(disp0, cam0.get_forward_vector()))
            
            op0 = obtainPixel(actor.get_transform(), cam0)
            # print("THEORETICAL POSITION IN CAM 0", actor, op0)
            
            cam0x.append(op0[0])
            cam0y.append(op0[1])
                
                
            # PIXEL COORD CALCULATIONS - SEE ABOVE METHOD
            
            resultantActors.append(str(actor))
            ids.append(actor.id)
            types.append(actor.type_id)
            distances.append(avgdist)
            angles.append(angle0)
            
            # Add the actor to a datapoint
        
            key = gaga.type_hash(actor.type_id)
            if (key == "n/a"):
                continue
                
            # print("ACTOR!", actor.type_id, actor.id)
            # print(key, str(actor.id), " = ", avgdist)
            data[key + str(actor.id)] = float(avgdist)
            
        if (DEBUG):
            print("VEHICLES")
            print(vehicle_transform.location)
            print(vehicle_transform.rotation)
            print(vehicle_forward)
            print("CAMERAS")
            # print(vehicle_matrix)
            # print(cam0_absolute)
            print(cam0, cam0.get_forward_vector())
            # print(cam1_absolute)
            print(cam1, cam1.get_forward_vector())
            # print(cam2_absolute)
            print(cam2, cam2.get_forward_vector())
            
        comp = (frame // framesperslide) * framesperslide
        
        place = str(save_path) + "/actordists/frame_" + ('%04d.txt' % comp)
        if (DEBUG):
            print("AUTOPILOT SAVING DATA %04d" % comp)
            print(place)
            
        if (frame % framesperslide == 0):
            savefile = open(place, "x")
        else:
            savefile = open(place, "a")
            
        # savefile.write("[FRAME " + str(frame) + "] ")
        # savefile.write(str(vehicle_transform) + "/" + str(vehicle_forward) + "\n")
        # savefile.write(str(cam0) + "/" + str(cam1) + "/" + str(cam2) + "\n")
        # savefile.write(str(cam0.get_forward_vector()) + "/" + str(cam1.get_forward_vector()) + "/" + str(cam2.get_forward_vector()) + "\n")
        for index in range(len(ids)):
            savefile.write(str(frame) + "/" + str(ids[index]) + "/" + str(types[index]) + "/" + str(distances[index]) + "/" + str(angles[index]))
            # savefile.write("[" + str(cam0x[index]) + " " + str(cam0y[index]))
            savefile.write("\n")
        savefile.close()
        
        
        
        
        tracker.update_graph(data, None)     
        rs = tracker.return_state()   
        gada.solve(frame, data, str(save_path) + "/actordists", rs, actors, cams)
    
    print(data)
    
    
    print("TOTAL TIME FOR DATA COLLECTION :", timeit.default_timer() - TIME_BEGIN)
    
        
        
        
        
        
        
        
