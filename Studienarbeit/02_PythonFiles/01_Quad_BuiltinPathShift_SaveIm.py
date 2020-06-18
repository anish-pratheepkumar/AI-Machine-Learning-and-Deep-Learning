# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 20:54:48 2020

@author: anish pratheepkumar

code to collect dataset for NN training:
Quadcopter model in CoppeliaSim follow path using inbuilt path function of CoppeliaSim
Collect images streamed by vision sesnor, categorise into classes and save in seperate folders
Classes : Forward Left (FL), Forward Right (FR) and Forward Straight (FS)
"""

#import essential libraries
import sim
import sys
import numpy as np
#import time
#import matplotlib.pyplot as plt
import cv2
import os 

#defining class data_collector to process vision_sensor image and then save the image in 
#folders specific to each class
class data_collector1:
    def __init__(self,lapcount):
        self.dic = {'FL':0, 'FR':0, 'FS1': 0, 'FS2': 0}
        self.cwd = os.getcwd()
        self.folder = 'NN_DATA_Raw/{}/'.format(lapcount)
        self.image_format = '.jpg'
     
    def process_image(self, resolution, image):
        image = np.array(image, dtype = np.uint8)                                   #signed_int -> unsigned int now each value range 0-255
        image.resize([resolution[0],resolution[1],3])                               #resize to 512*512*3
        image = np.flip(image,0)                                                    #flip the image w.r.t height, since image is read with origin at top now by flipping image is saved correctly
        #plt.imshow(image)                                                          #to display the image
        return image
        
    def save_image(self, heading, image):
        path = '{}{}/{}{}'.format(self.folder,heading,str(self.dic[heading]),self.image_format) #faster than the code below
        #path = self.folder + heading + '/' + str(self.dic[heading]) + self.image_format #path and image name  with image format included
        #print(path)        
        cv2.imwrite(os.path.join(self.cwd, path), image)
        self.dic[heading] += 1 


#main function for quadcopter motion control and data collection
def coppelia_sim(lapcount, class_name):
    sim.simxFinish(-1)   # just in case, close all opened connections
    
    clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)   #connect to CoppeliaSim
    
    #verify connection establlished with CoppeliaSim
    if clientID!=-1:
        print ('Connected to remote API server')
    else:
        print("Not connected to remote API server")
        sys.exit("Could not connect")
    
    
    
    #getting handle of dummy, quad and vision sensor
    err_code,target_handle = sim.simxGetObjectHandle(clientID,'Quadricopter_target',sim.simx_opmode_blocking)
    err_code,dummy_handle = sim.simxGetObjectHandle(clientID,'Dummy',sim.simx_opmode_blocking)
    err_code, vision_handle = sim.simxGetObjectHandle(clientID, 'Vision_sensor', sim.simx_opmode_blocking)
    
    #setting lapcount folder name (images will be saved in this folder)
    lapcount = 'Lap'+str(lapcount)
    data_collector = data_collector1(lapcount)   #creates an object of save image class (initialisations inside this class will be already done by this call)
    #print(lapcount)
    
    #initialise streaming of image data
    err_code, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_handle, 0, sim.simx_opmode_streaming)  #initialise to recieve the images
    
    #verify images are being recieved
    while True:
        err_code1, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_handle, 0, sim.simx_opmode_buffer)    #getting images
        if err_code1 == sim.simx_return_ok:
            break   
    
    #initialise prev_dummy_pos variable to zero for x,y and z coordinates
    prev_dummy_pos = [0.0,0.0,0.0]
           
    while True:
        #get dummy position & orientation
        err_code, dummy_orien = sim.simxGetObjectOrientation(clientID, dummy_handle, -1, sim.simx_opmode_blocking)    
        err_code, dummy_pos = sim.simxGetObjectPosition(clientID, dummy_handle, -1, sim.simx_opmode_blocking)
        dummy_pos[2] =  +5.0000e-01      #set a constant height of fligt
        
        #set position & orientation of the quadcopter
        err_code = sim.simxSetObjectPosition(clientID, target_handle, -1, dummy_pos, sim.simx_opmode_oneshot)
        err_code = sim.simxSetObjectOrientation(clientID, target_handle, -1, dummy_orien, sim.simx_opmode_oneshot)
        
        #define change in x and y coordinates
        delta_x = dummy_pos[0]-prev_dummy_pos[0]
        delta_y = dummy_pos[1]-prev_dummy_pos[1]
        
        if delta_x==0 or delta_y==0:            
            #save images of straight track          
            err_code1, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_handle, 0, sim.simx_opmode_buffer)    #getting images
            image = data_collector.process_image(resolution, image)     #accessing image processing function from data_collector class
            data_collector.save_image(class_name[0],image)                       #accessing save_inage function(of data_collector class) with arguments heading = FS(ForwardStraight) and image
                 
        else:
            #save images of curved (left or right) track          
            err_code1, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_handle, 0, sim.simx_opmode_buffer)    #getting images
            image = data_collector.process_image(resolution, image)     #accessing image processing function from data_collector class
            data_collector.save_image(class_name[1],image)                       #accessing save_inage function(of data_collector class) with arguments heading = FR(ForwardRight)/FL(ForwardLeft) and image

        #update the prev_dummy_pos variable
        prev_dummy_pos = dummy_pos
        
       
        #end the process when final position is reached
        if round(dummy_pos[0], 3) == 0.300 and  round(dummy_pos[1], 3) == -0.855: #pos[0]=0.126 for ACW motion(giving Fl images) & pos[0]=0.300 for CW motion(giving FR images)
            #stop streaming images
            err_code, resolution, image = sim.simxGetVisionSensorImage(clientID, vision_handle, 0, sim.simx_opmode_discontinue)
            err_code = sim.simxStopSimulation(clientID,sim.simx_opmode_oneshot_wait)            
            break
        #time.sleep(0.05)


#to run the code
if __name__ == "__main__" :
    lapcount = 'Final'                              #mention lap count 
    class_name = ['FS2','FR']                       #for CW motion of quad ['FS1','FL'] and for CW motion of quad ['FS2','FR']
    clientID,target_handle = coppelia_sim(lapcount, class_name) #run the quad in a lap numder 'lapcount' and save images in the corresponding lapcount folder




