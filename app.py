import pickle
import pandas as pd
import streamlit as st

# Load trained model
with open("laptop_price_model.pkl", "rb") as f:
    model = pickle.load(f)

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

    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {int(price):,}")
