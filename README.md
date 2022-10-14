# Brain Stroke Prediction
This project contains code for brain stroke prediction using public dataset, includes EDA, model training, and deploying using streamlit.

<p align="center">
  <img src="https://www.yashodahealthcare.com/blogs/wp-content/uploads/2021/08/Best-Doctor-or-Hospitals-for-Stroke-Brain-Attack-Treatment-in-Delhi-NCR-ghaziabad.jpeg" width=400 />
</p>

Streamlit App URL: https://samata18-brain-stroke-prediction-brain-stroke-deploy-394359.streamlitapp.com/

## Data used:
- Attribute Information
  * Gender :gender of the patient [M/F]
  * Age :age of the patient [years]
  * Hypertension : [yes / No]
  * Heart Disease  : [yes / No]
  * Ever married  : [yes / No]
  * Work Type : Work type of the patient wheather it is private job, gov job or self employed
  * Residence type : Residence type of the patient [Urban /Rural]
  * average glucose level :A fasting blood sugar level of 99 mg/dL or lower is normal, 100 to 125 mg/dL 
  * BMI : Body mass index (BMI) is a person's weight in kilograms divided by the square of height in meters. 
  * Smoking status :Smoking status of the patient
  * Stroke :  output class [1: heart disease, 0: Normal]

## Preprocessing Steps:
- Encoding Categorical Features
- Scaling The variance in Features
- Drop ID feature
- Check for nulls

## Modeling Algorithms:
- SVM
- Random Forest Classifier
- Logistic Regression

## Deployment:
- To make this application user friendly, it deployed by the help of streamlit library. Streamlit library is a library which helps to build application in minimal steps.


