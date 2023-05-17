import sqlite3

conn = sqlite3.connect('sign_coordinates.db')
print ('database opened')
cursor = conn.cursor()

def getdata():
    cursor.execute("SELECT * FROM DATA")
    for line in cursor.fetchall():
        print(line)

getdata()