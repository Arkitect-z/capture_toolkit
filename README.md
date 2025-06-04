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

### 1. Common Tactile Sensor Types
- **Resistive sensors**: Change resistance based on applied pressure
- **Capacitive sensors**: Change capacitance with touch/proximity
- **Piezoelectric sensors**: Generate voltage when deformed

### 2. Arduino-based Tactile Reading
```python
python utils/Optitrack2SMPL/demo/visualize.py
```
