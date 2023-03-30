import serial
import time

ser = serial.Serial("/dev/ttyUSB0", 115200)

def getTFminiData():
    counter = 0
    total_distance = 0
    while counter < 10:
        count = ser.in_waiting
        if count > 8:
            recv = ser.read(9)   
            ser.reset_input_buffer() 
            if recv[0] == 0x59 and recv[1] == 0x59:
                distance = recv[2] + recv[3] * 256
                strength = recv[4] + recv[5] * 256
                #print('(', distance, ',', strength, ')')
                ser.reset_input_buffer()
                total_distance += distance
            counter += 1
    return (total_distance / 10)
# Grab averge of 10 values distance value from getTFminiData()



if __name__ == '__main__':
    try:
        if ser.is_open == False:
            ser.open()
        d = getTFminiData()
        print(d)
        exit()
    except KeyboardInterrupt:   # Ctrl+C
        if ser != None:
            ser.close()