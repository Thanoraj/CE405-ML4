import streamlit as st
import numpy as np
import joblib

# Assuming your model and scaler are already saved in the specified paths
model_path = "model.pkl"
scaler_path = "scaler.pkl"

# Load your trained model and scaler
loaded_model = joblib.load(model_path)
loaded_scaler = joblib.load(scaler_path)


def predict_with_grnn(X_new):
    """Function to predict PAC using the loaded model and scaler."""
    X_scaled = loaded_scaler.transform(X_new)
    y_pred = loaded_model.predict(X_scaled)
    return y_pred


# Streamlit app layout
st.title("Water Quality to PAC Prediction (RAW water)")


# Create numerical input fields for input
turbidity = st.number_input(
    "Turbidity", min_value=0.0, max_value=368.0, value=28.0, format="%.2f"
)
ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, format="%.2f")
conductivity = st.number_input(
    "Conductivity", min_value=0.0, max_value=718.0, value=75.0, format="%.2f"
)
colour = st.number_input("Colour", min_value=0, max_value=1661, value=150, format="%d")
lime = st.number_input("Lime", min_value=0.0, max_value=4.0, value=1.65, format="%.2f")

# Button to make prediction
if st.button("Predict PAC"):
    new_data = np.array([[turbidity, ph, conductivity, colour, lime]])
    predicted_output = predict_with_grnn(new_data)
    predicted_pac_rounded = round(predicted_output[0], 1)

    st.write(f"Predicted PAC: {predicted_pac_rounded}")
