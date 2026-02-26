import streamlit as st

from labs.lab5_pca_autoencoder.pca_playground import render_pca_lab
from utils.ui import display_footer

st.set_page_config(
    page_title="Lab 5: PCA & Autoencoders",
    page_icon="🧠",
    layout="wide",
)

render_pca_lab()
display_footer()
