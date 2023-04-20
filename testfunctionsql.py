import sqlite3

conn = sqlite3.connect('testfunc.db')
print ('database opened')
cursor = conn.cursor()

# initialize new table to prevent duplicates (run only once) 
# TABLE DATA FORMAT
# SIGN TEXT
# COORDINATE TEXT (combine latitude and longitude before adding to database)
# DATE: TEXT
# TIME: TEXT (use hour/min data exclude seconds)
# SIGN COORDINATE AND TIME ARE UNIQUE TO A SPECIFIC POINT (TO PREVENT DUPLICATES AT THE SAME TIMESTAMP)

# initializes table in database
#conn.execute("CREATE TABLE DATA(SIGN TEXT, COORDINATE TEXT, DATE TEXT, TIME TEXT, unique(SIGN, COORDINATE, DATE, TIME));")

# functions for each command needed to add data to the table
# insert function
# create statement to execute
def insert(sign, coordinate, date, time):
    # all values are unique to duplicates can be ignored 
    cursor.execute("INSERT OR IGNORE INTO DATA(SIGN, COORDINATE, DATE, TIME) VALUES(?, ?, ?, ? );", (sign, coordinate, date, time))
    conn.commit()
    table = cursor.execute("SELECT * FROM DATA;")
    table = table.fetchall()
    if table[0] is not None:
        print("SIGN: ", table[-1][0], " COORDINATE: ", table[-1][1], " DATE: ", table[-1][2], " TIME: ", table[-1][3])
        print("records added")
    else: 
        print("insertion error")
    
# print function
# for entire table
def print_table():
    rows = cursor.execute("SELECT * FROM DATA;")
    for items in rows: 
        print("SIGN: ", items[0], " COORDINATE: ", items[1], " DATE: ", items[2], " TIME: ", items[3])

# test values
ts = 'stop'
tcord = '27493825.4328791, 43890214.38970125'
tdate = '01/23/23'
ttime = '11:32'

insert(ts, tcord, tdate, ttime)
#print_table()

ts = 'yield' 
tcord = '78942314.542, 80389271.541'
tdate = '02/22/23'
ttime = '23:45'
insert(ts, tcord, tdate, ttime)

insert('30mi', '472391.58019, 4830921.493701', '02/03/23', '11:33')

#print_table()

insert('30mi', '472391.58019, 4830921.493701', '02/03/23', '11:33')
#print_table()
insert('30mi', '472391.58019, 4830921.493701', '02/04/23', '11:33')

print_table()

#print_update()

conn.close()