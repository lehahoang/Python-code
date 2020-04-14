"""
    This source file will return the mean and std of the accuracy stored in each csv file.
    The result will be dumped to a txt file
    data structure of the input csv file>
    1st fault rate; 2nd: seed; 3rd: accuracy
    bash command: python lazy.py | tee luu.txt
"""
import numpy as np
import csv
import os


data_dir= '/homes/lhoang/ownCloud/Simulation results/CODES paper 2020/11-04-2020/resnet32/unbias/per-channel'# Path to original images
fn = os.listdir(data_dir) # load file names into a list
fn.sort() # sort the files in alphabet order

for file in fn:
    with open(data_dir+'/'+file, 'r') as reader:
        d=[]
        t=csv.reader(reader, delimiter=',')
        header=next(t) # skip the first row (header)
        for row in t:
            d.append(float(row[2])) # add the values to a list
        a=np.asarray(d, dtype=np.float32) # convert a list to a numpy array
        # print(file,': ', 'mean: {:.2f},std:{:.2f}'.format(a.mean(),a.std()))
        print(file,': ', '{:.2f}'.format(sum(d)/len(d)))
