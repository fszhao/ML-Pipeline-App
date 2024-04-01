import streamlit as st
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import pandas as pd


st.set_page_config(
    page_title="Model Comparison",
    page_icon="",
)

# Retrieve the data from session state
df = st.session_state.df

st.title("Examine Your Dataset")

#file = st.sidebar.file_uploader("Upload Dataset Here")

# if file:
#     df = pd.read_csv(file)
#     df.to_csv("sourcedata.csv", index_col=None)
#     st.dataframe(df)
    
#     st.title("Data Profile")
#     profile_report = df.profile_report()
#     st_profile_report(profile_report)

st.dataframe(df)
st.title("Data Profile")
profile_report = df.profile_report()
st_profile_report(profile_report)