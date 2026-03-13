import streamlit as st

from labs.lab7_classification.classification_lab import render_classification_lab
from utils.ui import display_footer

st.set_page_config(
    page_title="Lab 7: The Wolf of Wall Street",
    page_icon="📈",
    layout="wide",
)

render_classification_lab()
display_footer()
