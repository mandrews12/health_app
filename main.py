import streamlit as st
import pandas as pd

# compute 95% confidence intervals for classification and regression
# problems

def classification_confint(acc, n):
    '''
    Compute the 95% confidence interval for a classification problem.
      acc -- classification accuracy
      n   -- number of observations used to compute the accuracy
    Returns a tuple (lb,ub)
    '''
    import math
    interval = 1.96*math.sqrt(acc*(1-acc)/n)
    lb = max(0, acc - interval)
    ub = min(1.0, acc + interval)
    return (lb,ub)

st.write("Hello world!")

# Input form
age = st.slider('Age', 18, 100, 45)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure (trestbps)', 80, 200, 120)
chol = st.slider('Serum Cholestoral (chol)', 100, 600, 240)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.selectbox('Resting ECG Results (restecg)', [0, 1, 2])
thalach = st.slider('Max Heart Rate Achieved (thalach)', 60, 220, 150)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
