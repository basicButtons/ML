import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

with open("NewTrain.csv" ,"r") as f:
    reader = csv.reader(f)
    i = 0
    temp=[]
    res = []
    for row in reader:
        i =  i+1
        if 1 < i < 5:
            temp = temp + row[0:5]
            
        if i == 5:
            temp = temp+row
            res.append(temp)
            temp = temp[5:len(temp)-1]

        if i > 5:
            temp =temp + row
            res.append(temp)
            temp = temp[5:len(temp)-1]

        
with open("/Users/maxuan/Desktop/course/ML/HW2/data/res.csv","w") as f:
    writer = csv.writer(f)
    writer.writerows(res)



