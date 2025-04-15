import streamlit as st
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import firestore

# Set page configuration first
st.set_page_config(layout="wide")

# Initialize Firebase
if not firebase_admin._apps:
    creds = service_account.Credentials.from_service_account_info(st.secrets["firebase"])
    firebase_admin.initialize_app(creds, {
        'projectId': 'diabetes-predictor-f16bb'
    })

db = firestore.client()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

data = load_data()

# Preprocess
X = data.drop(columns='Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# UI
st.title('🩺 Diabetes Prediction App')
st.markdown("Enter patient details in the sidebar to predict diabetes.")

st.sidebar.header('Patient Details')

# Sidebar inputs
pregnancies = st.sidebar.number_input('Pregnancies', 0, 20, 1)
glucose = st.sidebar.number_input('Glucose Level', 0, 200, 120)
bp = st.sidebar.number_input('Blood Pressure', 0, 150, 70)
skin_thickness = st.sidebar.number_input('Skin Thickness', 0, 100, 20)
insulin = st.sidebar.number_input('Insulin Level', 0, 900, 30)
bmi = st.sidebar.number_input('BMI', 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
age = st.sidebar.number_input('Age', 0, 120, 30)

# Predict button
if st.sidebar.button('Predict'):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    result_text = '⚠️ The person is likely to have diabetes.' if prediction[0] == 1 else '✅ The person is not likely to have diabetes.'
    st.success(result_text)

    # Generate numeric ID for the document
    predictions_ref = db.collection('predictions').stream()
    count = sum(1 for _ in predictions_ref) + 1  # Get the next available numeric ID
    doc_id = str(count)  # Use the count as the ID

    # Save to Firestore
    db.collection('predictions').document(doc_id).set({
        'pregnancies': pregnancies,
        'glucose': glucose,
        'bp': bp,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'dpf': dpf,
        'age': age,
        'prediction_result': result_text,  # Add prediction result
        'date': datetime.datetime.now().strftime("%d-%m-%Y")
    })

# Show Firestore Prediction History
if st.checkbox('📋 Show Prediction History'):
    st.subheader('Stored Predictions')

    all_preds = db.collection('predictions').stream()
    prediction_list = [doc.to_dict() | {'id': doc.id} for doc in all_preds]

    if prediction_list:
        df = pd.DataFrame(prediction_list)
        df = df[['id', 'pregnancies', 'glucose', 'bp', 'skin_thickness', 'insulin', 'bmi', 'dpf', 'age', 'prediction_result', 'date']]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions stored yet.")

# Show Raw Dataset
if st.checkbox('🗃️ Show Raw Diabetes Dataset'):
    st.subheader('Diabetes Dataset')
    st.dataframe(data.head())

# Show specific record by ID and allow to update or delete
if st.checkbox('🔍 View and Update/Delete Prediction Record'):
    record_id = st.text_input("Enter Record ID to Update or Delete", "")
    
    if record_id:
        # Fetch the selected record
        selected_record = db.collection('predictions').document(record_id).get()

        if selected_record.exists:
            selected_record_data = selected_record.to_dict()

            # Display fields for updating
            st.write(f"**Prediction Result:** {selected_record_data.get('prediction_result', 'N/A')}")
            st.write(f"**Pregnancies:** {selected_record_data.get('pregnancies', 'N/A')}")
            st.write(f"**Glucose Level:** {selected_record_data.get('glucose', 'N/A')}")
            st.write(f"**Blood Pressure:** {selected_record_data.get('bp', 'N/A')}")
            st.write(f"**Skin Thickness:** {selected_record_data.get('skin_thickness', 'N/A')}")
            st.write(f"**Insulin Level:** {selected_record_data.get('insulin', 'N/A')}")
            st.write(f"**BMI:** {selected_record_data.get('bmi', 'N/A')}")
            st.write(f"**Diabetes Pedigree Function:** {selected_record_data.get('dpf', 'N/A')}")
            st.write(f"**Age:** {selected_record_data.get('age', 'N/A')}")
            st.write(f"**Date:** {selected_record_data.get('date', 'N/A')}")
            
            # Option to delete the record
            delete = st.button(f"Delete Record {record_id}")
            if delete:
                db.collection('predictions').document(record_id).delete()
                st.success(f"Record {record_id} deleted successfully.")
            
            # Option to update the record
            st.subheader(f"Update Record {record_id}")
            new_prediction_result = st.text_input("Enter new Prediction Result", value=selected_record_data.get('prediction_result', ''))
            new_pregnancies = st.number_input("Pregnancies", value=selected_record_data.get('pregnancies', 0))
            new_glucose = st.number_input("Glucose Level", value=selected_record_data.get('glucose', 0))
            new_bp = st.number_input("Blood Pressure", value=selected_record_data.get('bp', 0))
            new_skin_thickness = st.number_input("Skin Thickness", value=selected_record_data.get('skin_thickness', 0))
            new_insulin = st.number_input("Insulin Level", value=selected_record_data.get('insulin', 0))
            new_bmi = st.number_input("BMI", value=selected_record_data.get('bmi', 0.0))
            new_dpf = st.number_input("Diabetes Pedigree Function", value=selected_record_data.get('dpf', 0.0))
            new_age = st.number_input("Age", value=selected_record_data.get('age', 0))
            
            # Update button
            if st.button(f"Update Record {record_id}"):
                db.collection('predictions').document(record_id).update({
                    'pregnancies': new_pregnancies,
                    'glucose': new_glucose,
                    'bp': new_bp,
                    'skin_thickness': new_skin_thickness,
                    'insulin': new_insulin,
                    'bmi': new_bmi,
                    'dpf': new_dpf,
                    'age': new_age,
                    'prediction_result': new_prediction_result  # Update prediction result
                })
                st.success(f"Record {record_id} updated successfully.")

        else:
            st.write("No such record found!")
