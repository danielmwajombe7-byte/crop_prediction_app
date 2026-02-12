import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Function to train the model directly from the CSV file
@st.cache_resource
def load_and_train():
    # Load the dataset already uploaded to GitHub
    df = pd.read_csv('multi_crop_yield_data.csv')
    
    # Preprocess the data
    le = LabelEncoder()
    df['Crop'] = le.fit_transform(df['Crop'])
    
    X = df[['Temperature', 'Rainfall', 'Soil_pH', 'Crop']]
    y = df['Yield']
    
    # Train the RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le

# Initialize model and encoder
model, le = load_and_train()

# Streamlit UI Configuration
st.set_page_config(page_title="Crop Prediction", page_icon="🌾")
st.title("🌾 Crop Yield Prediction System")

# Input fields for the user
temp = st.number_input("Enter Temperature (°C)", value=25.0)
rain = st.number_input("Enter Rainfall (mm)", value=100.0)
ph = st.number_input("Enter Soil pH", value=6.5)
crop_type = st.selectbox("Select Crop Type", le.classes_)

# Prediction logic
if st.button("Predict Yield"):
    crop_encoded = le.transform([crop_type])[0]
    prediction = model.predict([[temp, rain, ph, crop_encoded]])
    st.success(f"Estimated Yield: {prediction[0]:.2f} Tons")
