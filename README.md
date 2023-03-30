# computer_vision_module

This repository contains the code for our senior capstone project, Computer Vision Module with Road Sign and Obstacle Detection.  

main.py contains the main file which when run, obtains camera frame and classifies what appears in the object.  It also displays the percent accuracy and the distance from the center of the device the object is from in meters.  

lidar_averaging.py contains the lidar averaging function.  The function obtains 10 readings from the lidar using the serial port and averages the 10 readings.  It then returns the average of these readings, this is done in order to get more accurate results from the lidar.  