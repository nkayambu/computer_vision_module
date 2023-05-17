
import time
import board
import busio

import adafruit_gps



import serial
uart = serial.Serial("/dev/ttyTHS1", baudrate=9600, timeout=10)




gps = adafruit_gps.GPS(uart, debug=False)  


gps.send_command(b"PMTK314,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0")
gps.send_command(b"PMTK220,1000")


def getGPSdata():
    last_print = time.monotonic()
    while True:
        gps.update()
        
        current = time.monotonic()
        if current - last_print >= 1.0:
            last_print = current
            if not gps.has_fix:
                
                print("Waiting for fix...")
                continue

            latitude = "{0:.3f} degrees".format(gps.latitude)
            longitude = "{0:.3f} degrees".format(gps.longitude)
            date = "{}/{}/{}".format(gps.timestamp_utc.tm_mon,  
                    gps.timestamp_utc.tm_mday,  
                    gps.timestamp_utc.tm_year,)
            tim = "{:02}:{:02}".format(gps.timestamp_utc.tm_hour, gps.timestamp_utc.tm_min)
            #break
            return latitude, longitude, date, tim

latitude, longitude, date, tim = getGPSdata()
print(latitude + " " + longitude + " " + date + " " + tim)