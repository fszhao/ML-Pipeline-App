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

st.title("Model Comparison")
target = st.selectbox("Select Target Variable", df.columns)

setup(df, target=target, silent = True)
setup_df = pull()
st.info("ML Experiment Settings")
st.dataframe(setup_df)
best_model = compare_models()
compare_df = pull()
st.info("Performance of Different Models")
st.dataframe(compare_df)
best_model
save_model(best_model, 'best_model')