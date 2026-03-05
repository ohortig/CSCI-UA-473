import streamlit as st

from labs.lab6_k_means_clustering.kmeans_lab_student import render_kmeans_lab
from utils.ui import display_footer

st.set_page_config(
    page_title="Lab 6: K-Means Clustering",
    page_icon="🎯",
    layout="wide",
)

render_kmeans_lab()
display_footer()
