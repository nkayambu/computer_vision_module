# computer_vision_module

This repository contains the code for our senior capstone project, Computer Vision Module with Road Sign and Obstacle Detection.  

main.py contains the main file which when run, obtains camera frame and classifies what appears in the object.  It also displays the percent accuracy and the distance from the center of the device the object is from in meters.  

lidar_averaging.py contains the lidar averaging function.  The function obtains 10 readings from the lidar using the serial port and averages the 10 readings.  It then returns the average of these readings, this is done in order to get more accurate results from the lidar.  

detection_w_classifier.py contains the new detection software.  This file will work similarly to main.py however will be implementing the sliding windows in order to detect multiple objects.  This will be considered the new "main" software in this project and will be running alongside the lidar, testfunctionssql, gps, and detection_helpers. 

detection_helpers.py contains the functions to implement the sliding window and image pyramid to work for object detection.  

testfunctionssql.py contains the functions to insert and print different values from the database.  

database_initializer.py contains code that only needs to be run once to create the database and create a table to hold data in that database.  The file specifies the different data being collected and how it is being stored.  Below is an overview of what is being stored in the database.  

Database: 
Table Data
>ALL DATA IS STORED AS TEXT
>SIGN
>LATITIUDE
>LONGITUDE
>DATE
>TIME
>ALL VALUES ARE UNIQUE FOR DUPLICATE HANDLING