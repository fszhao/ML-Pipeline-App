import streamlit as st
import pandas as pd
import os 
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model


# Page Config
st.set_page_config(
    page_title="Data Ingestion",
    page_icon="",
    
)

st.title("Welcome!")
st.image('images/robo.jpg')



# Check if you've already initialized the data
if 'data' not in st.session_state:
    st.session_state['input_data.csv'] = ""
    # Prompt User to Enter Data
    st.write("A dataset is required to get started.")


file = st.file_uploader("Upload Your Own Dataset Here:")

if os.path.exists("input_data.csv"):
    df = pd.read_csv(file)
    df.to_csv("input_data.csv", index_col=None)
    st.dataframe(df)
    st.session_state.df = df
else:
    st.write("By default, we will use the UCI Wisconsin Breast Cancer Dataset")
    df = pd.read_csv('data/wisc_bc_data.csv')
    #st.dataframe(df)
    st.session_state.df = df

    

