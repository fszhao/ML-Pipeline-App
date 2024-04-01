import streamlit as st
import pandas as pd
import os 

from pycaret.classification import setup, compare_models, pull, save_model


# Page Config
st.set_page_config(
    page_title="Model Evaluation",
    page_icon="",
)

# Retrieve the data from session state
df = st.session_state.df

st.title("Download the Best Performing Model")
# with open("best_model.pkl", 'rb') as f:
#     st.download_button("Download Model File", f, "best_model.pkl")