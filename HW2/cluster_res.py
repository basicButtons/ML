from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train_data = pd.DataFrame(pd.read_csv("/Users/maxuan/Desktop/course/ML/HW2/data/res.csv"))
training = np.array(train_data)
km = KMeans(n_clusters=4)
km = km.fit(training[:,0:20])
labels = km.labels_
print(km.cluster_centers_)

a = np.array([0])
b = np.array([1])
c = np.array([2])
d = np.array([3])

training_0 = []
training_1 = []
training_2 = []
training_3 = []

for number,label in zip(training,labels):
    number = number.tolist()
    if label == 0:
        training_0.append(number)
    elif label == 1:
        training_1.append(number)
    elif label ==2:
        training_2.append(number)
    else:
        training_3.append(number)

import csv
test_first=[]
test_second=[]
test_third=[]
test_fourth = []
with open("/Users/maxuan/Desktop/course/ML/HW2/data/test.csv") as f:
    reader = csv.reader(f)
    for row in reader:
        if km.predict([row[0:20]]) == a:
            test_first.append(row)
        if km.predict([row[0:20]]) == b:
            test_second.append(row)
        if km.predict([row[0:20]]) == c:
            test_third.append(row)
        if km.predict([row[0:20]]) == d:
            test_fourth.append(row)


with open("/Users/maxuan/Desktop/course/ML/HW2/data/train0.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(training_0)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/train1.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(training_1)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/train2.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(training_2)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/train3.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(training_3)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/test_0.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(test_first)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/test_1.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(test_second)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/test_2.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(test_third)
with open("/Users/maxuan/Desktop/course/ML/HW2/data/test_3.csv","w") as f:
    writer=csv.writer(f)
    writer.writerows(test_fourth)