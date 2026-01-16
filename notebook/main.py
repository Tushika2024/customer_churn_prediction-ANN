import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

# --- PAGE CONFIG ---
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

# --- STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    ##load the trained model
    model=tf.keras.models.load_model('notebook\model.h5')
    ##loading encoder and scalar
    with open(r'D:\all-ml-projects\ann-project\notebook\label_encoder_gender.pkl','rb') as file:
        label_encoder=pickle.load(file)
    with open(r'D:\all-ml-projects\ann-project\notebook\one_hot_enocder_geo.pkl','rb') as file:
        one_hot_encoder=pickle.load(file)
    with open(r'D:\all-ml-projects\ann-project\notebook\scaler.pkl','rb') as file:
        scaler=pickle.load(file)
    return model,label_encoder,one_hot_encoder,scaler

model, label_encoder, one_hot_encoder, scaler = load_assets()

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2415/2415053.png", width=100)
st.sidebar.title("Configuration")
st.sidebar.info("This ANN model predicts if a customer will leave the bank based on their profile.")

# --- MAIN UI ---
st.title("🏦 Customer Churn Prediction")
st.write("Fill in the details below to analyze customer retention.")
st.divider()

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

# --- DISPLAY RESULTS ---
st.subheader("Prediction Result")
# Progress Bar for Churn Probability
st.write(f"Churn Probability:")
st.progress(float(prediction_proba))
if prediction_proba > 0.5:
    st.error(f"### 🚩 HIGH RISK: {prediction_proba:.2%}")
    st.markdown('<div class="prediction-box" style="background-color: #ffcccc; color: #cc0000;">Customer Likely to Churn</div>', unsafe_allow_html=True)
else:
    st.success(f"### ✅ LOW RISK: {prediction_proba:.2%}")
    st.markdown('<div class="prediction-box" style="background-color: #d4edda; color: #155724;">Customer Likely to Stay</div>', unsafe_allow_html=True)

# Show data used for prediction
with st.expander("View Input Feature Vector"):
    st.dataframe(input_df) 
