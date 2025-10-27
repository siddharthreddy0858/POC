import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="EV Battery Health Predictor (POC)",
    page_icon="🔋",
    layout="wide"
)

# --- 2. Load the Trained Model ---
MODEL_FILE = 'ev_battery_soh_predictor.joblib'

@st.cache_resource
def load_model(model_path):
    """Loads the trained XGBoost model."""
    if not os.path.exists(model_path):
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_FILE)

# --- 3. Sidebar for Inputs (Simulated API) ---
st.sidebar.title("⚙️ Simulated API Input")
st.sidebar.markdown("""
This sidebar simulates the data that would be fetched from a client's API. 
In the final product, the user would only enter a VIN.
""")

# --- Vehicle Specification Inputs ---
st.sidebar.header("Vehicle Specifications")
battery_capacity_kwh = st.sidebar.number_input(
    "Battery Capacity (kWh)", 
    min_value=2.0, max_value=120.0, value=30.2, step=0.1,
    help="Original nominal capacity of the vehicle's battery pack (e.g., 30.2 for a Nexon EV)."
)

# This is no longer needed for the new RUL calculation, but we'll leave it
# as it might be useful for other models.
vehicle_efficiency_wh_km = st.sidebar.number_input(
    "Vehicle Efficiency (Wh/km)", 
    min_value=50.0, max_value=300.0, value=130.0, step=1.0,
    help="Average energy consumed per kilometer (e.g., 130 for a Nexon EV)."
)

battery_chemistry = st.sidebar.selectbox(
    "Battery Chemistry", 
    ('NMC', 'LFP'),
    help="The chemical composition of the battery."
)

# --- Live BMS Data Inputs ---
st.sidebar.header("Live BMS Data")

# --- UPDATED: Sliders changed to number_input (text boxes) ---
odometer_km = st.sidebar.number_input(
    "Odometer (km)", 
    min_value=0, max_value=1000000, value=45000, step=1000
)

cycle_count = st.sidebar.number_input(
    "Cycle Count", 
    min_value=0, max_value=5000, value=550, step=10
)

fast_charges = st.sidebar.number_input(
    "Total Fast Charges", 
    min_value=0, max_value=5000, value=42, step=5
)

avg_temp_c = st.sidebar.number_input(
    "Average Operating Temperature (°C)", 
    min_value=-10.0, max_value=50.0, value=28.0, step=0.5,
    help="The vehicle's average operating temperature over its lifetime."
)
# --- End of Update ---


# --- 4. Main Application Interface ---
st.title("🔋 EV Battery Health & RUL Predictor (POC)")

# Explanation Box for your manager
with st.expander("ℹ️ How this Proof of Concept (POC) works", expanded=True):
    st.markdown("""
    This application demonstrates the power of our predictive model.

    * **In the Final Product:** A user will only enter their **VIN**.
    * **Our System Will Then:**
        1.  Call the client's API to get the vehicle's latest data (odometer, cycle counts, etc.).
        2.  Feed that data into our trained model.
        3.  Display the predicted health and remaining range.
    
    For this POC, the **sidebar on the left simulates the API's job**. Use it to manually provide the vehicle data and see the model's prediction in real-time.
    """)

if model is None:
    st.error(f"**Error:** Model file '{MODEL_FILE}' not found.")
    st.markdown(f"""
    Please make sure the trained model file (`{MODEL_FILE}`) is in the
    same directory as this Streamlit `app.py` file.
    """)
else:
    # --- 5. Prediction Logic ---
    if st.button("Predict Battery Health", type="primary"):
        
        # 1. Pre-process inputs
        chem_NMC = 1 if battery_chemistry == 'NMC' else 0
        chem_LFP = 1 if battery_chemistry == 'LFP' else 0
        
        # 2. Create the input DataFrame
        features_list = [
            'Odometer_km', 'Cycle_Count', 'Fast_Charges', 'Avg_Temp_C',
            'Battery_Capacity_kWh', 'chem_LFP', 'chem_NMC'
        ]
        input_data = {
            'Odometer_km': odometer_km, 'Cycle_Count': cycle_count,
            'Fast_Charges': fast_charges, 'Avg_Temp_C': avg_temp_c,
            'Battery_Capacity_kWh': battery_capacity_kwh,
            'chem_LFP': chem_LFP, 'chem_NMC': chem_NMC
        }
        input_df = pd.DataFrame([input_data], columns=features_list)
        
        # 3. Make the SOH prediction
        predicted_soh = model.predict(input_df)[0]
        predicted_soh = np.clip(predicted_soh, 0, 100) 
        
        # --- 4. NEW, CORRECTED RUL CALCULATION ---
        TARGET_SOH = 80.0
        
        # SOH remaining until target
        remaining_soh_pct = predicted_soh - TARGET_SOH
        
        # SOH lost so far
        soh_lost_pct = 100.0 - predicted_soh
        
        remaining_range_km = 0.0
        
        if remaining_soh_pct <= 0:
            remaining_range_km = 0.0
        elif soh_lost_pct <= 0 or odometer_km == 0:
            # If no degradation or no mileage, we can't extrapolate.
            # This implies a brand new car, so RUL is effectively infinite or unknown.
            remaining_range_km = -1 # We'll use -1 as a code for "N/A"
        else:
            # This is the new, correct logic:
            # 1. Calculate the historical degradation rate (% SOH lost per km)
            degradation_rate_per_km = soh_lost_pct / odometer_km
            
            # 2. Extrapolate: (Remaining SOH %) / (SOH % lost per km) = Remaining km
            remaining_range_km = remaining_soh_pct / degradation_rate_per_km
        
        
        # --- 6. Display the Results ---
        
        # Custom message for the RUL
        if remaining_range_km == 0.0:
            st.subheader("✅ This vehicle's battery is at or below the service threshold.")
        elif remaining_range_km == -1:
            st.subheader("✅ This vehicle is brand new. Insufficient data to predict remaining range.")
        else:
            st.subheader(f"✅ This vehicle can run for an estimated **{remaining_range_km:,.0f} km** more.")
        
        st.caption("This is the estimated distance until the battery's State of Health (SOH) reaches 80% and may require service.")
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Predicted State of Health (SOH)",
                value=f"{predicted_soh:.2f} %"
            )
            st.progress(float(predicted_soh) / 100)

        with col2:
            if remaining_range_km >= 0:
                st.metric(
                    label="Estimated Remaining Range",
                    value=f"{remaining_range_km:,.0f} km",
                    help=f"Calculated based on a target SOH of {TARGET_SOH}%."
                )
            else:
                 st.metric(
                    label="Estimated Remaining Range",
                    value="N/A",
                    help="Insufficient history to extrapolate remaining range."
                )

        # Display the replacement message
        if predicted_soh <= 80.5:
            st.error(f"**Action Recommended:** Battery is at or below the {TARGET_SOH}% health threshold. Please schedule a service for inspection or replacement.", icon="🚨")
        elif predicted_soh <= 85.0:
            st.warning(f"**Notice:** Battery SOH is approaching the {TARGET_SOH}% health threshold. We recommend planning for a future battery service.", icon="⚠️")
        else:
            st.success("**Status:** Battery is in good health.", icon="👍")

