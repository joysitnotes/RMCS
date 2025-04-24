# RMCS_CAM
Remote Mobile CCTV System for Red Teamers

A simple Flask CCTV system.

# IPCAMERA:
Hardware:
  -  Raspberry Pi 0 W 2
  -  4G LTE Module
  -  NV Camera
  -  Power Bank

Software:
  -  IPCAM/app.py

# C2 Server:
Hardware:
  - PC
  - External Storage (USB drive, SSD, HDD)
Software:
 - RMCS_CAM/app.py


# Setup
## RTSPCAMERA: Raspberrypi 0 w 2:
change creadentials in .yml
run RTSPCAM/setup.sh on raspberrypi 0  w 2 to setup rtsp camera

## CAMERA SERVER
Create a Pushbullet account and add API key to RMCS_CAM/APPS/describe.py to enable notifications

run pip install -r requirements.txt to install libraries
run python3 app.py to start server
