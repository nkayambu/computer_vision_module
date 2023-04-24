# RUN THIS CODE ONLY ONCE TO INITIALIZE THE DATABASE AND CREATE THE TABLE

import sqlite3
# create database
conn = sqlite3.connect('sign_coordinates.db')
print ('database opened')
cursor = conn.cursor()


# initializes table in database
# TABLE DATA
# ALL DATA IS STORED AS TEXT
# SIGN
# LATITIUDE
# LONGITUDE
# DATE
# TIME
# ALL VALUES ARE UNIQUE FOR DUPLICATE HANDLING
conn.execute("CREATE TABLE DATA(SIGN TEXT, LATITUDE TEXT, LONGITUDE TEXT, DATE TEXT, TIME TEXT, unique(SIGN, LATITUDE, LONGITUDE, DATE, TIME));")


conn.close()