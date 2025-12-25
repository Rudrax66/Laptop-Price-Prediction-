import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# =========================
# CONFIG
# =========================
MODEL_FILE = "laptop_model.pkl"

st.set_page_config(page_title="Laptop Price Prediction", layout="centered")


# =========================
# SAFE MODEL LOADER
# =========================
def load_model_safely():
    try:
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error("❌ Model load failed")
        st.exception(e)
        st.stop()


model = load_model_safely()


# =========================
# UI
# =========================
st.title("💻 Laptop Price Prediction")

ram = st.number_input("RAM (GB)", 4, 64, step=4)
ram_type = st.selectbox("RAM Type", ["DDR3", "DDR4", "DDR5"])

rom = st.number_input("Storage (GB)", 128, 2048, step=128)
rom_type = st.selectbox("Storage Type", ["SSD", "HDD"])

gpu = st.selectbox("GPU", ["Integrated", "NVIDIA", "AMD"])
display_size = st.number_input("Display Size (inch)", 10.0, 18.0)

res_w = st.number_input("Resolution Width", 1024, 3840)
res_h = st.number_input("Resolution Height", 768, 2160)

os_name = st.selectbox("Operating System", ["Windows", "Mac OS", "Linux"])
warranty = st.selectbox("Warranty", ["Yes", "No"])
spec_rating = st.slider("Spec Rating", 1, 10)


# =========================
# PREDICTION
# =========================
if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "ram": ram,
        "ram_type": ram_type,
        "rom": rom,
        "rom_type": rom_type,
        "gpu": gpu,
        "display_size": display_size,
        "resolution_width": res_w,
        "resolution_height": res_h,
        "os": os_name,
        "warranty": warranty,
        "spec_rating": spec_rating
    }])

    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")
    except Exception as e:
        st.error("❌ Prediction failed")
        st.exception(e)

