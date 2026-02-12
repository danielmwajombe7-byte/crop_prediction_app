import streamlit as st
import joblib
import numpy as np

# Load model and encoder
model = joblib.load("crop_yield_model.pkl")
encoder = joblib.load("crop_encoder.pkl")

st.set_page_config(page_title="🌾 Crop Yield Prediction", layout="wide")

st.title("🌾 Crop Yield Prediction System (Tanzania)")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.header("Enter Crop Data")
    temperature = st.number_input("Temperature (°C)", 10.0, 45.0)
    rainfall = st.number_input("Rainfall (mm)", 30.0, 400.0)
    soil_ph = st.number_input("Soil pH", 4.5, 8.0)
    
    available_crops = list(encoder.classes_)
    crop = st.selectbox("Crop Type", available_crops)
    
    if st.button("Predict Yield"):
        # Encode crop
        crop_encoded = encoder.transform([crop])[0]
        input_data = np.array([[temperature, rainfall, soil_ph, crop_encoded]])
        
        # Predict yield
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)  # prevent negative yield
        
        # Store prediction in session state to show in right column
        st.session_state['prediction'] = prediction
        st.session_state['selected_crop'] = crop

with col2:
    st.header("Prediction Result")
    if 'prediction' in st.session_state:
        st.success(f"🌱 Estimated Yield for {st.session_state['selected_crop']}: {st.session_state['prediction']:.2f} tons")
        
        # Compare all crops
        st.subheader("Predicted yields for all known crops:")
        yields = {}
        for c in available_crops:
            c_encoded = encoder.transform([c])[0]
            y = model.predict(np.array([[temperature, rainfall, soil_ph, c_encoded]]))[0]
            y = max(0, y)
            yields[c] = y
            st.info(f"{c}: {y:.2f} tons")
        
        # Best crop
        best_crop = max(yields, key=yields.get)
        st.balloons()
        st.success(f"🌟 Best crop for current conditions: {best_crop} → {yields[best_crop]:.2f} tons")
    else:
        st.info("Enter crop data on the left and click 'Predict Yield' to see results.")
