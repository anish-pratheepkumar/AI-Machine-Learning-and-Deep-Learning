# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:54:48 2020

@author: anish pratheepkumar

code for quadcopter model in CoppeliaSim to follow track uisng 
CoppeliaSim inbuilt path function.
"""
#import essential libraries
import sim
import sys

sim.simxFinish(-1) # just in case, close all opened connections

clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) #connect to CoppeliaSim

#verify that connection is established with coppeliasim
if clientID!=-1:
    print ('Connected to remote API server')

else:
    print("Not connected to remote API server")
    sys.exit("Could not connect")

#get the handles of quadcopter target and the dummy
err_code,target_handle = sim.simxGetObjectHandle(clientID,'Quadricopter_target',sim.simx_opmode_blocking)
err_code,dummy_handle = sim.simxGetObjectHandle(clientID,'Dummy',sim.simx_opmode_blocking)

while True:
    #get the position and orientation of the dummy
    err_code, dummy_orien = sim.simxGetObjectOrientation(clientID, dummy_handle, -1, sim.simx_opmode_blocking)    
    err_code, dummy_pos = sim.simxGetObjectPosition(clientID, dummy_handle, -1, sim.simx_opmode_blocking)
    dummy_pos[2] =  +5.0000e-01  #keeping the height of quadcopter constant
    
    #set the position and orientaion of the quadcopter target
    err_code = sim.simxSetObjectPosition(clientID, target_handle, -1, dummy_pos, sim.simx_opmode_oneshot)
    err_code = sim.simxSetObjectOrientation(clientID, target_handle, -1, dummy_orien, sim.simx_opmode_oneshot)
    
    #break the loop when quad reaches the final position
    if round(dummy_pos[0], 3) == 0.300 and  round(dummy_pos[1], 3) == -0.855:   ##pos[0]=0.126 for left & =0.3 for right
        err_code = sim.simxStopSimulation(clientID,sim.simx_opmode_oneshot_wait)
        break

