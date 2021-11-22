"""
Task 4
author: Céline Bitter
date: 29.05.19
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical
from keras.optimizers import SGD
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Import the data  
train_labeled = pd.read_hdf("train_labeled.h5", "train")
train_unlabeled = pd.read_hdf("train_unlabeled.h5", "train")
test = pd.read_hdf("test.h5", "test")

x_train_labeled = train_labeled.drop(["y"], 1)
x_train_labeled = x_train_labeled.values
print(x_train_labeled.shape)

x_train_unlabeled = train_unlabeled.values
print(x_train_unlabeled.shape)

x_test = test.values

y_train_labeled = train_labeled["y"]
y_train_labeled = y_train_labeled.values.copy()
y_train_labeled = to_categorical(y_train_labeled, 10).copy()
print(y_train_labeled.shape)

# Normalize your data
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
x_train_unlabeled = scaler.transform(x_train_unlabeled)

# Train the neural network 
model = Sequential()

model.add(Dense(units=1000, input_shape = (x_train.shape[1],), activation='relu'))
model.add(Dropout(0.5))
    
model.add(Dense(units=500, activation='relu'))
model.add(Dropout(0.5))
    
model.add(Dense(units=y_train.shape[1], activation='softmax'))
sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size = 32)

classes = model.predict_classes(x_test, batch_size=128)

test["y"] = list(classes)
test["Id"] = test.index
y_test = test[["Id", "y"]]

y_test.to_csv("results_celine.csv", header = True, index = False)

