import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression


# =========================
# FILE PATHS
# =========================
DATA_FILE = "laptop_data.csv"
MODEL_FILE = "laptop_price_model.pkl"


# =========================
# CLEANING FUNCTIONS
# =========================
def clean_ram(val):
    val = str(val).upper().strip()
    return int(val.replace("GB", ""))


def clean_rom(val):
    val = str(val).upper().strip()
    if "TB" in val:
        return int(float(val.replace("TB", "")) * 1024)
    elif "GB" in val:
        return int(float(val.replace("GB", "")))
    else:
        return np.nan


# =========================
# TRAIN & SAVE MODEL
# =========================
def train_and_save_model():
    df = pd.read_csv(DATA_FILE)

    # standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # drop useless cols
    df.drop(columns=["name", "unnamed: 0", "unnamed: 0.1"], inplace=True, errors="ignore")

    # clean ram / rom
    df["ram"] = df["ram"].apply(clean_ram)
    df["rom"] = df["rom"].apply(clean_rom)

    FEATURE_COLS = [
        "ram",
        "ram_type",
        "rom",
        "rom_type",
        "gpu",
        "display_size",
        "resolution_width",
        "resolution_height",
        "os",
        "warranty",
        "spec_rating"
    ]

    X = df[FEATURE_COLS]
    y = df["price"]

    cat_cols = [
        "ram_type",
        "rom_type",
        "gpu",
        "os",
        "warranty"
    ]

    num_cols = [
        "ram",
        "rom",
        "display_size",
        "resolution_width",
        "resolution_height",
        "spec_rating"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(pipe, f)

    return pipe


# =========================
# LOAD OR TRAIN MODEL
# =========================
if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
else:
    model = train_and_save_model()


# =========================
# STREAMLIT UI
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

    price = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹ {int(price):,}")
