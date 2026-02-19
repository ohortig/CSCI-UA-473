import streamlit as st

from labs.lab4_generalization_data_splits.generalization_lab import (
    render_generalization_lab,
)
from utils.ui import display_footer

# Page Config
st.set_page_config(
    page_title="Lab 4: Generalization & Data Splits",
    page_icon="🔀",
    layout="wide",
)

# Render Lab
render_generalization_lab()

# Footer
display_footer()
