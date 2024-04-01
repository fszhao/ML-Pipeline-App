import streamlit as st
import pandas as pd
import os 

# Profiling
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.classification import setup, compare_models, pull, save_model

if os.path.exists("sourcedata.csv"):
    df = pd.rerad_csv("sourcedata.csv", index_col=None)

with st.sidebar:
    #st.image()
    st.title("Machine Learning Model Comparitor")
    choice = st.radio("Navigation", ["Data Evalution", "Model Comparison", "Model Download"])

if choice == "Data Evalution":
    st.title("File Upload for Modeling")
    file = st.file_uploader("Upload Dataset Here")
    if file:
        df = pd.read_csv(file)
        df.to_csv("sourcedata.csv", index_col=None)
        st.dataframe(df)
        
        st.title("Data Profile")
        profile_report = df.profile_report()
        st_profile_report(profile_report)


if choice == "Model Comparison":
    st.title("Model Comparison")
    target = st.selectbox("Select Target Variable", df.columns)
    if st.button("Train Model"):
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

if choice == "Model Download":
    st.title("Download the Best Performing Model")
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download Model File", f, "best_model.pkl")
