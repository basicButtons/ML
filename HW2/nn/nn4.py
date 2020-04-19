import csv
import numpy as np
from sklearn import preprocessing
train_x = []
train_y = []
test_x = []
test_y = []

with open("/Users/maxuan/Desktop/course/ML/HW2/data/train3.csv",'r') as f:
    reader = csv.reader(f)
    for row in reader:
        train_x.append(row[0:20])
        train_y.append(row[20])

with open("/Users/maxuan/Desktop/course/ML/HW2/data/test_3.csv",'r') as f:
    reader = csv.reader(f)
    for row in reader:
        test_x.append(row[0:20])
        test_y.append(row[20])

train_y_1 = np.array(train_y).reshape(len(train_y),1)
test_y_1 = np.array(test_y).reshape(len(test_y),1)
train_y_1 = train_y_1.astype(float)
test_y_1 = test_y_1.astype(float)
print(test_y_1.dtype)

train_x = np.array(train_x)
test_x_1 = np.array(train_x)
x_MinMax = preprocessing.MinMaxScaler()
train_x_1 = x_MinMax.fit_transform(train_x)
test_x_1 = x_MinMax.fit_transform(test_x)
print(train_x_1.dtype)

from sklearn.neural_network import MLPRegressor
fit1 = MLPRegressor(hidden_layer_sizes=(60,100), activation='relu',solver='adam',alpha=0.07,max_iter=600)
fit1.fit(train_x_1,train_y_1)
pred1_test = fit1.predict(test_x_1)

from sklearn.metrics import mean_squared_error
mse_1 = mean_squared_error(pred1_test,test_y_1)
rmse_1 = pow(mse_1,0.5)
print(mse_1)
print(1-(rmse_1/49.5))