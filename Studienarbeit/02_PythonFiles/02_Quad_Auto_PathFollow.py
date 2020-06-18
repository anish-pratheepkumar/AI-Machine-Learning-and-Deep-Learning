# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:12:47 2020

@author: anish pratheepkumar

code to run the quadcopter model autonomously on a track using trained CNN model
"""
#import essential libraries
import sim
import sys
#import os
#import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
from keras.models import load_model

#Loading trained CNN model with weights
PATH = '/home/anish/anaconda_py3_copelia'
QuadNet = load_model(PATH + '/Model/Quad_Net_Wt.h5')
time.sleep(0.5) 

#just in case, close all opened connections to CoppeliaSim
sim.simxFinish(-1)
#connect to CoppeliaSim
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)
#verify that connection is established with coppeliasim
if clientID!=-1:
    print ('Connected to remote API server')

else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")


#getting object handles for quad control
err_code,target_handle = sim.simxGetObjectHandle(clientID,'Quadricopter_target',sim.simx_opmode_blocking)

 
#initialise var for accessing LUA function at Server
inputInts=[]                  #dtype table (inside it string)
inputFloats=[]                #dtype table (inside it floats)
inputStrings=[]               #dtype table inside it strings
inputBuffer=''                #dtype stirng
   



while True:

    ###Getting Image using LUAcode in server(coppeliaSim))###        
    res,retTable1,retTable2,retTable3,retString=sim.simxCallScriptFunction(clientID,'Vision_sensor',sim.sim_scripttype_childscript,
                        'getImage',inputInts,inputFloats,inputStrings,inputBuffer,sim.simx_opmode_blocking) 
    if res==sim.simx_return_ok:
        image = retString
        resolution = retTable1
    
    #Image Processing
    image = np.array(image, dtype = np.uint8)                                   #signedint -> unsigned int now each value range 0-255
    image.resize([resolution[0],resolution[1],3])                               #resize to 512*512*3
    image = np.flip(image,0)
    image = cv2.resize(image,(int(256/2),int(256/2)))                           #resize image to model input dimension 128x128
    image = image[None,:,:,:]
    
    #using QuadNet to predict the quad motion    
    y_pred = QuadNet.predict(image)
    cls_pred = np.argmax(y_pred,axis=1)
    cls = np.squeeze(cls_pred)
    #print (cls)
    
    #getting current pos & orien of the quad
    err_code, target_orien_body = sim.simxGetObjectOrientation(clientID, target_handle, target_handle, sim.simx_opmode_blocking)
    err_code, target_pos_body = sim.simxGetObjectPosition(clientID, target_handle, target_handle, sim.simx_opmode_blocking)
            
    #condtion for motion control of the quad (setting pos&orien based on QuadNetwork prediction)
    if cls == 0:
        #move Left
        target_pos_body[0] = target_pos_body[0] + (0.018)
        target_orien_body[2] = target_orien_body[2] + 0.02618
        
        err_code = sim.simxSetObjectOrientation(clientID, target_handle, target_handle, target_orien_body, sim.simx_opmode_oneshot)
        err_code = sim.simxSetObjectPosition(clientID, target_handle, target_handle, target_pos_body, sim.simx_opmode_oneshot)
        
    elif cls == 1:
        #move Right
        target_pos_body[0] = target_pos_body[0] + (0.018)
        target_orien_body[2] = target_orien_body[2] - 0.0349
        
        err_code = sim.simxSetObjectOrientation(clientID, target_handle, target_handle, target_orien_body, sim.simx_opmode_oneshot)
        err_code = sim.simxSetObjectPosition(clientID, target_handle, target_handle, target_pos_body, sim.simx_opmode_oneshot)
        
    else:
        #move forward
        target_pos_body[0] = target_pos_body[0] + (0.018)
        target_orien_body[2] = target_orien_body[2] + 0.0
        
        err_code = sim.simxSetObjectOrientation(clientID, target_handle, target_handle, target_orien_body, sim.simx_opmode_oneshot)
        err_code = sim.simxSetObjectPosition(clientID, target_handle, target_handle, target_pos_body, sim.simx_opmode_oneshot)
    #time.sleep(0.025) 
           
        

