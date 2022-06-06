# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 00:59:21 2022

@author: Giga Hertz
"""

import numpy as np
import pickle

loaded_model = pickle.load(open('E:/Python & JDK/Deploying Machine Learning model/cardio.sav', 'rb'))

input_data = (130,0,4.89,25.98,0,72,30.42,14.71,23)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Coronary Heart Disease')
else:
  print('The Person has a Coronary Heart Disease')