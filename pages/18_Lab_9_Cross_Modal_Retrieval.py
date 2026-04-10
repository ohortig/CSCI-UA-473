import streamlit as st

from labs.lab9_cross_modal_retrieval.crossmodal_lab_student import render_crossmodal_lab
from utils.ui import display_footer

st.set_page_config(
    page_title="Lab 9: Cross-Modal (Instructor)",
    page_icon="🔗",
    layout="wide",
)

render_crossmodal_lab()
display_footer()
