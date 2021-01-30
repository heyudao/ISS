# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:55:06 2021

@author: He Yudao
"""
import numpy as np
import h5py

with h5py.File('model.h5','r') as hdf:
    ls = list(hdf.keys())
    print('List of datasets in this file: \n', ls)
    data = hdf.get('conv2d')
    dataset1 = np.array(data)
    
    print('Shape of dataset1: \n', dataset1)
    
f = h5py.File('model.h5','r')
ls = list(f.keys())
print(dataset1)
f.close()

###########################


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
print(model.summary())

