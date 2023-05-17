import cv2
import numpy as np
from PIL import Image
#import argparse
##import imutils
import time
import tensorflow as tf
from lidar_averaging import getTFminiData
from GPS import getGPSdata
from testfunctionsql import insert
import sqlite3
import signal

conn = sqlite3.connect('sign_coordinates.db')
print ('database opened')
cursor = conn.cursor()

# database backup 
f = open("data.txt", "w")


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException
signal.signal(signal.SIGALRM, timeout_handler)

# dummy gps values

latitude = "1.2345" 
longitude = "5.4321" 
date = "1/1/2023" 
ti = "12:59"



CATEGORIES = ["10 speed limit sign", "15 speed limit sign", "25 speed limit sign", "30 speed limit sign", "35 speed limit sign", "45 speed limit sign", "Stop Sign", "Unknown"]


# load model
model = tf.keras.models.load_model("7_class_unknown.model")

# video
video = cv2.VideoCapture(0)


#video.set(cv2.CAP_PROP_FPS, 10)

while True:
    time.sleep(1)
    ret, frame = video.read()
    #if (ret == False):
    #    print("No camera")
    #    break

    # convert image to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)


    # resize for input to model 256x256
    im = im.resize((256, 256))
    img_array = np.array(im)

    # change input to input tensor
    img_array = np.expand_dims(img_array, axis=0)

    # predict method
    prediction = model.predict(img_array)
    prediction_class = np.argmax(prediction)
    print(f"class: {prediction_class}, score: {prediction[0][prediction_class]}")
    if prediction[0][prediction_class] > 0.65:
        p = CATEGORIES[prediction_class]
        s = prediction[0][prediction_class]
        print(p)
        print(prediction_class)
        distance = getTFminiData()
        for i in range(3):
            signal.alarm(3)
            try:
                latitude, longitude, date, ti = getGPSdata()
            except TimeoutException:
                continue
            else:
                signal.alarm(0)
        ps = str(CATEGORIES[prediction_class])
        print("Sign: " + ps + " Latitude: " + latitude + " Longitude: " + longitude + " Date: " + date + " Time: " + ti + " Distance: " + str(distance) + "cm")
        f.write(ps + " " + latitude + " " + longitude + " " + date + " " + ti + "\n")
        if ps is "Unknown":
            continue
        else: 
            #try:
            insert(ps, latitude, longitude, date, ti)
            #except sqlite3.Error:
            #    continue
        distance = str(distance) + " cm"
        text = str(p) + str(s)
        org = (50,50)
        #sorg = (50, 100)
        dorg = (50, 100)
        #cv2.putText(frame, text, org, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,0,0))
        #cv2.putText(frame, s, sorg, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,0,0))
        #cv2.putText(frame, distance, dorg, fontFace = cv2.FONT_HERSHEY_COMPLEX, fontScale = 1, color = (250,0,0))
        
        

    # if no input
    #if prediction:
    #    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("Capturing", frame)
    key=cv2.waitKey(1)
    if key == ord('q'):
        break
conn.close()

video.release()
cv2.destroyAllWindows()
