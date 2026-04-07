import streamlit as st

from labs.lab10_rl.rl_lab_teacher import render_rl_lab
from utils.ui import display_footer

st.set_page_config(
    page_title="Lab 10: Reinforcement Learning",
    page_icon="🤖",
    layout="wide",
)

render_rl_lab()
display_footer()
