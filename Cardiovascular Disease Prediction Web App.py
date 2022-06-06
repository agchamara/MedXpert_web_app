# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 08:19:26 2022

@author: Chamara
"""

import numpy as np
import pickle
import streamlit as st

#Load model
loaded_model = pickle.load(open('E:\AI project/cardio.sav','rb'))

#funtion for prediction

def cardio_detect(input_data):
    input_data_as_numpy_array= np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0]== 0):
      print('The Person does not have a Coronary Heart Disease')
    else:
      print('The Person has a Coronary Heart Disease')
      

def main():
    
    # Giving a title 
    st.title('Cardiovascular Disease Predictor')
    
    input_data = (130,0,4.89,25.98,0,72,30.42,14.71,23)
    
    sbp = st.text_input('Systolic Blood Pressure Value')
    tobacco = st.text_input('Cumulative Tobacco Quantity(kg)')
    ldl = st.text_input('Low Density Lipoprotein Cholesterol Level')
    adiposity = st.text_input('Adiposity Value')
    famhist = st.text_input('Family History of Heart Disease ("Absent" = 0 | "Present" = 1)')
    typea = st.text_input('Type-A Behavior')
    obesity = st.text_input('Obesity Value')
    alcohol = st.text_input('Current Alcohol Consumption Value')
    age = st.text_input('Age at Onset')
    
    
    # Code for prediction
    detect = ''
    
    # Creting a button for prediction
    
    if st.button('Predict Cardiovascular Disease'):
        detect = cardio_detect([sbp,tobacco,ldl,adiposity,famhist,typea,obesity,alcohol,age])
        
    st.success(detect)
    
   
    
if __name__ == '__main__':
    main()