import streamlit as st

st.set_page_config(
    page_title="HVAC-Analysis",
    layout="wide",
    page_icon="htms_logo.jpg")

    # Page header
col1, col2 = st.columns([5,1])
#with col1:
with col2:
        st.image("htms_logo.jpg", width=150)

from apps import app_mini_hvac, app_plant_hvac

option = st.selectbox(
    "Choose Analysis",
    ("Mini-HVAC Analysis", "Plant-HVAC Analysis")
)

if option == "Mini-HVAC Analysis":
    app_mini_hvac.run()
else:
    app_plant_hvac.run()
