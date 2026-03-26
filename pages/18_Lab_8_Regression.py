import streamlit as st

from labs.lab8_regression.regression_lab_student import render_regression_lab
from utils.ui import display_footer

st.set_page_config(
    page_title="Lab 8: Regression",
    page_icon="📈",
    layout="wide",
)

render_regression_lab()
display_footer()
