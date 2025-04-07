import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm

# Load the diabetes dataset
@st.cache_data
def load_data():
    data = pd.read_csv("diabetes.csv")
    return data

data = load_data()

# Split data
X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Streamlit interface
st.title('Diabetes Prediction App')

st.sidebar.header('Enter Patient Details')

# Input fields
pregnancies = st.sidebar.number_input('Number of Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input('Glucose Level', min_value=0, max_value=200, value=120)
bp = st.sidebar.number_input('Blood Pressure', min_value=0, max_value=150, value=70)
skin_thickness = st.sidebar.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.sidebar.number_input('Insulin Level', min_value=0, max_value=900, value=30)
bmi = st.sidebar.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5)
age = st.sidebar.number_input('Age', min_value=0, max_value=120, value=30)

# Make prediction
if st.sidebar.button('Predict'):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.write('The person is likely to have diabetes.')
    else:
        st.write('The person is not likely to have diabetes.')
