import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

##load the trained model
model=tf.keras.models.load_model('notebook\model.h5')

##loading encoder and scalar
with open(r'D:\all-ml-projects\ann-project\notebook\label_encoder_gender.pkl','rb') as file:
    label_encoder=pickle.load(file)
with open(r'D:\all-ml-projects\ann-project\notebook\one_hot_enocder_geo.pkl','rb') as file:
    one_hot_encoder=pickle.load(file)
with open(r'D:\all-ml-projects\ann-project\notebook\scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    

st.title("Customer Churn Prediction")
st.write("Enter the customer details below to predict the likelihood of churn.")

credit_score = st.number_input("Credit Score")
geography = st.selectbox("Geography",one_hot_encoder.categories_[0])
gender = st.selectbox("Gender", label_encoder.classes_)
age = st.slider("Age", 18, 100)
tenure = st.slider("Tenure (Years)", 0, 10)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_products = st.number_input("Number of Products", 1, 4)
has_crcard = st.selectbox("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
is_active = st.selectbox("Is Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

input_dict = {
    'CreditScore': credit_score,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_products,
    'HasCrCard': has_crcard,
    'IsActiveMember': is_active,
    'EstimatedSalary': salary
}
input_df = pd.DataFrame([input_dict])
input_df['Gender']=label_encoder.transform(input_df['Gender'])
geo_encoded=one_hot_encoder.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=one_hot_encoder.get_feature_names_out(['Geography']))

input_df=pd.concat([input_df.reset_index(drop=True),geo_encoded_df],axis=1)

input_scaled=scaler.transform(input_df)

predicted=model.predict(input_scaled)
prediction_proba=predicted[0][0]
st.write(f"prediction prob: {prediction_proba}")
if(prediction_proba>0.5):
    st.write("Customer likely to churn")
else:
    st.write("Customer is not likely to churn")
    
