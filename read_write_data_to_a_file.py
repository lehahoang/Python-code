'''
This script performs the following works:
@ write data to a .csv file
@ Read data from .csv file
@ Compute the mean and std of set of data taken from .csv file
'''
import csv
import os
import numpy as np
read_data_file =True
# The header will be written if the above condition is False. This condition should be set for the initialization of the data file
# Otherwise, data from the data file will be read
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# This source code below allow user to write/read data to/from the .csv file in the format below
# data category 1    data category 2   ...... data category n
#  value 1			    value 2        ......   value n
#     . 				  .			   ......	  .
#     . 				  .			   ......	  .
#
# 'value 1' to 'value n' are arranged in the column 1 to column n respectively
# Data preparation is shown below
header =[
        'data category 1',
        'data category 2',
        ....
        'data category n'
] # The first row of the .csv file

info = [
        'value 1',  # data category 1
        'value 2',  # data category 2
		...
        'value n',  # data category n
] # The next row of the .csv file which contains information

# Writing the data to the .csv file for the first time. File 'info.csv' is created from this point
# All pre-defined data will be erased completely   
    with open('./info.csv','w') as info_file: # w stands for write
        writer = csv.writer(info_file,delimiter=',')
        writer.writerow(header) # the header is written on the first row
		writer.writerow(info1)  # the data info1 is written on the second row
		.....
		writer.writerow(infon)	# the data infon is written on the nth+1 row

# Writing the data to the .csv file for the next time
# All previous data will be kept  
    with open('./info.csv','a') as info_file: # a stands for append
        writer = csv.writer(info_file,delimiter=',')
		writer.writerow(info1)  # the data info1 is written on the second row
		.....
		writer.writerow(infon)	# the data infon is written on the nth+1 row

# Reading the data from the .csv file
    with open('./info.csv','r') as info_file: # r stands for read
        reader=csv.reader(info_file,delimiter=',')
        for row in reader:
            print(row) # Print all the data from the first row to the final row
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# Writing data to a csv file as a dictionary
# This source code below allow user to write/read data to/from the .csv file in the format below
# {'key 1': value 1,
#  'key 2': value 2,
#  .............
#  'key n': value n
# }
# 
keys=[
	'key1',	
	'key2',
	....
	'keyn',
]
with open('./info1.csv','w') as info_file:
    writer     = csv.DictWriter(info_file,fieldnames=keys)
    writer.writeheader()
    writer.writerow({'key1':value1,
                     'key1' :value2,
		   .....
	    	   'keyn' :valuen					   
})

with open('./info1.csv','r') as info_file:
    reader=csv.DictReader(info_file,delimiter=',')
    for row in reader:
	print(row)

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

if not read_data_file:
# Writing data to the .csv file
else:
# Reading data to the .csv file

########################________________________######################
'''
    This source code dumps all needed parameters to a csv. file
    First, it creates a file and the header field of the file
    Second, it dumps all needed parameters row-by-row

'''


header=['Seed','Fault rate','Accuracy','Threshold','layer_name']

seed=0
fr=0.00000001
acc=83.2
th=99
layer_='third layer'
data_=[] # Data computed to have mean or std

if not os.path.isfile('save.csv'):
    with open('save.csv', 'wb') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(header)

with open('save.csv', 'a') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow([str(seed), str(fr), str(acc), str(th), layer_])
# The outcome looks like:
# Seed,Fault rate,Accuracy,Threshold,layer_name
# 1,0.0001,96.1,None,first layer
# 1,0.0001,96.1,None,first layer
# 52,0.0001,96.1,None,first layer
# 34,0.03,34.2,12.3,first layer
#......

with open('save.csv', 'r') as csvfile:
    filereader = csv.reader(csvfile)
    row = list(filereader)
    # print len(row) # Number of rows
    # print len(row[0]) # Number of columns
    for i in range(len(row)): # A walk throuh the first column then print out the value on the 3rd column
        if row[i][2]!='Accuracy':
            data_.append(float(row[i][2]))
    print(np.mean(data_))
    print(np.std(data_))
