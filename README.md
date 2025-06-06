# Multi-Sensor Data Reading Guide

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

You can download the data through [data-Google Drive](https://drive.google.com/file/d/1iLZRwSk4tO-tLo2YsVI2Y8X6j828dt7L/view?usp=drive_link) and [SMPL-Google Drive](https://drive.google.com/file/d/1MT74651uUVoZ3wGgFw_Lh19sVXoz6sgq/view?usp=drive_link) or simply run

```bash
# Prepare data
gdown 1iLZRwSk4tO-tLo2YsVI2Y8X6j828dt7L
unzip data_and_models.zip
mv data_and_models/* . && rmdir data_and_models/
# Prepare SMPL
gdown 1MT74651uUVoZ3wGgFw_Lh19sVXoz6sgq
unzip Optitrack2SMPL.zip -d utils/
```

## RealSense

### 1. Installation
1. Install [Intel RealSense Viewer](https://www.intelrealsense.com/sdk-2/) to simply visualize the data:
![RealSense Viewer Interface](Misc/EgoView_Screenshot.png)

2. Install package: `pip install pyrealsense2==2.51.1.4348`, tested successfully on python 3.9

### 2. Export data
The following script will export RGB, depth and IMU data shown above simultaneously.
```bash
python utils/realsense.py
```

## OptiTrack
There are 2 **flaws** in the existing code:
1. The current code can not handle new human skeleton and articulated object markers correctly, so the following scripts uses demo data instead.
2. The matching code for hand motion and object position is not complete.
### 1. Export data
```python
python utils/Optitrack2SMPL/demo/visualize.py
```

## Tactile


