# Project Title

* Autonomous Track Navigation of a Quadcopter using Deep Convolutional Neural Networks: Simulation-based evaluation 

## Operating System

* Main OS - Ubuntu 16.04 LTS (Xenial Xersus)
* Windows 10 (only for solidworks 3Dmodel)

## Softwares and tools

01. Anaconda Distribution 4.8.1
02. CoppeliaSim 4.0.0 
03. h5py 2.10.0
04. keras 2.3.1
05. matplotlib 3.2.1
06. numpy 1.17.3
07. opencv-python 4.2.0.32 
08. pandas 0.25.2
09. Python 3.5.0
10. scipy 1.4.1
11. seaborn 0.10.0
12. SolidWorks 2020
13. spyder 3.3.6

## Description for each directories

### 01_SolidWorksModels

  * consists of 3D models of Track in '.SLDPRT' and '.STL' format

### 02_CoppeliaSimScenes

1. scenes for collecting data:

  - 01_Quad_Manual_DataCollection_Left.ttt
  - 02_Quad_Manual_DataCollection_Right.ttt

2. scenes for testing the trained CNN model:

  - 03_Quad_Auto_PathFollow_TrainTrack.ttt
  - 04_Quad_Auto_TestTrack_1.ttt
  - 05_Quad_Auto_TestTrack_2.ttt

### 03_PythonFiles

Please Install CoppeliaSim 4.0.0 software from this [link](https://www.coppeliarobotics.com/downloads)

All python file should be in a single directory along with the following files for CoppeliaSim Remote API binding:
* sim.py
* simConst.py
* remoteApi.so
* simpleTest.py (optional, contains example)


1. File for implementing quad model path follow:

  - 00_Quad_BuiltinPath_Follow.py

running instructions:

* open CoppeliaSim scene - 01_Quad_Manual_DataCollection_Left.ttt
* open python file - 00_Quad_BuiltinPath_Follow.py
* run CoppeliaSim scene first and then the python file.
(dont wait too long after running simulation scene since the dummy in the scene starts to move with simulation start)

2. File for data collection:

  - 01_Quad_BuiltinPath_SaveIm.py

Inside the directory create folders for data collection:
Folder Heirarchy -

```bash
├── NN_DATA_Raw
│   └── Lap1
│       ├── FL
│       ├── FR
│       ├── FS1
│       └── FS2
```
similarly create lap counts(lap number) as needed, here only one lap shown. 

running instructions:

* for forward Left class images (ACW motion of the quad)
  - open CoppeliaSim scene - 01_Quad_Manual_DataCollection_Left.ttt
  - open python file - 01_Quad_BuiltinPath_SaveIm.py
  - Input for python file: lapcount(lap number) and class_name (['FS1','FL'])
  - run CoppeliaSim scene first and then the python file. (dont wait too long after running simulation scene since the dummy in the scene starts to move with simulation start)
  - This will save Forward Left and Forward Straight images to corresponding lap count directory

* for forward Right class images (CW motion of the quad)
  - open CoppeliaSim scene - 02_Quad_Manual_DataCollection_Right.ttt
  - open python file - 01_Quad_BuiltinPath_SaveIm.py
  - Input for python file: lapcount(lap number) and class_name (['FS2','FR'])
  - run CoppeliaSim scene first and then the python file. (dont wait too long after running simulation scene since the dummy in the scene starts to move with simulation start)
  - This will save Forward Right and Forward Straight images to corresponding lap count directory
  
3. File for autonomous path follow:

  - 02_Quad_Auto_PathFollow.py

running instructions:

* open CoppeliaSim scenes - 03_Quad_Auto_PathFollow_TrainTrack.ttt
* open python file - 02_Quad_Auto_PathFollow.py
* run the CoppeliaSim scene first and then the python file. (After running once for a Scene, restart the python kernel before running the file for next CoppeliaSim scene)

### 04_NN_Training_and_Evaluation

This directory consists of Jupyter notebook files for network design, fine-tuning and evaluation

1. File to create H5py dataset files from collected data set

  - 01_CreateDataSet-128x128.ipynb

2. File to train the CNN model and save model with trained weights

  - 02_NNTraining_KerasModel.ipynb

3. File for eavluation of the model on test set

  - 03_ConfusionMat_Cls Report.ipynb
  
### 05_Model

* consists of HDF5 file with model+trained weights
