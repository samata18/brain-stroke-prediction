# from matplotlib.animation import AVConvFileWriter
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st

def load_model(model_path):
    return joblib.load(model_path)

def preprocessing(df, ):
    preprocessing_dict = joblib.load("preprocessing_columns.pkl")
    for key, operation in preprocessing_dict.items():
        df[key] = operation.transform(df[key])
    return df

st.title("Brain Stroke Prediction")
st.markdown("Enter the required info and assess your risk of getting a brain stroke.")
gender = st.selectbox('Whats your gender?',
                    ('Male', 'Female', ))
age = st.slider('Please enter your age', 0, 130, 25)
hypertension = int(st.checkbox('History of Hypertension?'))
heart_disease = int(st.checkbox('History of Heart Disease?'))
ever_married = st.checkbox('Ever Married?')
ever_married = "Yes" if ever_married else "No"
work_type = st.selectbox('Work Type?', ('Private', 'Self-employed', 'Govt_job', 'children'))
Residence_type = st.selectbox('Residence Type?', ('Urban', 'Rural'))
avg_glucose_level = st.slider('Average Glucose Level', 10, 300, value=100, step=1)
bmi = st.slider('BMI', 10, 50, value=30, step=1)
smoking_status = st.selectbox('Smoking Status?', ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))

pred_instance = {'age':age, 'gender':gender, 'hypertension':hypertension,
                 'heart_disease': heart_disease,
                 'ever_married':ever_married, 'work_type': work_type,
                 'Residence_type': Residence_type, 'avg_glucose_level': avg_glucose_level,
                 'bmi': bmi, 'smoking_status': smoking_status}
pred_instance = pd.DataFrame(pred_instance, index=[0])
pred_instance = preprocessing(pred_instance)

model = load_model('forest_tuned.pkl')
st.button("Run Model")
prediction_proba = model.predict_proba(pred_instance)
# print(prediction_proba)
prediction = f"Chance of a stroke: {prediction_proba[0][1]*100:.2f}%"
# prediction = "Stroke Unlikely" if prediction == 0 else "Stroke Risk Present"
st.write("Result: " +prediction)