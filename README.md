# Dataset Toolkit

This document provides instructions for operating data from OptiTrack motion capture systems, Intel RealSense cameras, and tactile sensors.

## Table of Contents

1. [Overview](#overview)
2. [RealSense](#realsense)
3. [OptiTrack](#optitrack-setup)
4. [Tactile](#tactile-sensor-setup)

## Overview

This guide covers the integration and data reading from three types of sensors:

- **RealSense**: Intel depth cameras for **ego-view** RGB-D data acquisition
- **OptiTrack**: Motion capture system for human and objects
    - **Manus**: Hand motion capture gloves
- **Tactile Sensors**: Hand pressure sensors

You can download the data through [data-Google Drive](https://drive.google.com/file/d/1dTlTYgb09jW77nK7T4ALG2cZ0-bebG2O/view?usp=sharing) and [SMPL-Google Drive](https://drive.google.com/file/d/1gNyCf2G9gKQxY6I9Ydg3q8tw0rPwIWTF/view?usp=sharing) or simply run

### 1. Installation
```bash
git clone https://github.com/Arkitect-z/capture_toolkit.git && cd capture_toolkit
conda create -y -n capture python=3.9 && conda activate capture
pip install torch torchvision
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install -r requirements.txt
# pyrealsense2==2.51.1.4348 was tested successfully on python=3.9
# Prepare data
gdown 1dTlTYgb09jW77nK7T4ALG2cZ0-bebG2O
unzip data_pilot.zip && rm data_pilot.zip
# Prepare SMPL
gdown 1gNyCf2G9gKQxY6I9Ydg3q8tw0rPwIWTF
unzip Optitrack2SMPL.zip && rm Optitrack2SMPL.zip
```

## RealSense

### 1. Visualization
Install [Intel RealSense Viewer](https://www.intelrealsense.com/sdk-2/) to simply visualize the data:
![RealSense Viewer Interface](EgoView_Screenshot.png)

### 2. Export data
The following script will export RGB, depth and IMU data shown above simultaneously.
```bash
python utils/run_realsense.py
```

## OptiTrack
There are 2 **flaws** in the existing code:
1. ~~Current code can handle human skeleton and articulated object markers correctly, the error in the demo is due to a problem with the Optitrack system export.~~
2. The code for matching hand motion, tactile and object position is incomplete.
### 1. Export data
```bash
python utils/run_optitrack.py
```
This will export a video and SMPL motion data.

## Tactile


