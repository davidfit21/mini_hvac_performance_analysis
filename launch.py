import streamlit as st
import subprocess

col1, col2 = st.columns([5,1])
with col1:
    st.title("HVAC Analysis Launcher")
with col2:
    st.image("htms_logo.jpg", width=150)

st.set_page_config(page_title="HVAC Launcher", layout="wide", page_icon="htms_logo.jpg")
option = st.selectbox("Choose Analysis:", ["Mini-HVAC Analysis", "Plant-HVAC Analysis"])

if st.button("Run"):
    if option == "Mini-HVAC Analysis":
        st.info("Launching Mini-HVAC Analysis...")
        subprocess.Popen(["streamlit", "run", "app_mini_hvac.py"])
    elif option == "Plant-HVAC Analysis":
        st.info("Launching Plant-HVAC Analysis...")
        subprocess.Popen(["streamlit", "run", "app_plant_hvac.py"])