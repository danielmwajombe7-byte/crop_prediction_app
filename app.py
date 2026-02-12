import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --------------------------
# Load model and encoder
# --------------------------
model = joblib.load("crop_yield_model.pkl")  # Your trained LinearRegression model
encoder = joblib.load("crop_encoder.pkl")    # LabelEncoder with only trained crops

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(
    page_title="🌾 Crop Yield Predictor",
    page_icon="🌱",
    layout="wide"
)

# --------------------------
# Title
# --------------------------
st.title("🌾 Crop Yield Prediction System (Tanzania)")
st.markdown("Enter the details below to predict crop yield and see which crop is best for your conditions.")

# --------------------------
# Sidebar for inputs
# --------------------------
with st.sidebar:
    st.header("Input Parameters")

    temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=45.0, value=25.0, step=0.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=30.0, max_value=400.0, value=100.0, step=1.0)
    soil_ph = st.number_input("Soil pH", min_value=4.5, max_value=8.0, value=6.5, step=0.1)

    crop = st.selectbox("Crop Type", encoder.classes_)

# Encode crop
crop_encoded = encoder.transform([crop])[0]

# --------------------------
# Prediction Button
# --------------------------
if st.button("Predict Yield"):
    # Prepare input for model
    input_df = pd.DataFrame([[temperature, rainfall, soil_ph, crop_encoded]],
                            columns=["Temperature", "Rainfall", "Soil_pH", "Crop_Encoded"])

    try:
        prediction = model.predict(input_df)[0]

        # Show main result
        st.success(f"🌱 Estimated Crop Yield for {crop}: {prediction:.2f} tons")

        # --------------------------
        # Compare all crops for this condition
        # --------------------------
        st.markdown("### 📊 Best Crop for Current Conditions")
        yields = {}
        for c in encoder.classes_:
            c_encoded = encoder.transform([c])[0]
            temp_df = pd.DataFrame([[temperature, rainfall, soil_ph, c_encoded]],
                                   columns=["Temperature", "Rainfall", "Soil_P"] + ["Crop_Encoded"][-1:])
            y = model.predict(pd.DataFrame([[temperature, rainfall, soil_ph, c_encoded]],
                                          columns=["Temperature", "Rainfall", "Soil_P", "Crop_Encoded"]))[0]
            yields[c] = y

        # Find best crop
        best_crop = max(yields, key=yields.get)
        best_yield = yields[best_crop]
        st.info(f"✅ Best crop for these conditions: {best_crop} → {best_yield:.2f} tons")

        # Display all crop yields in a table
        st.markdown("### 📄 Predicted Yields for All Crops")
        result_df = pd.DataFrame(list(yields.items()), columns=["Crop", "Predicted Yield (tons)"])
        st.dataframe(result_df.style.format({"Predicted Yield (tons)": "{:.2f}"}))

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
