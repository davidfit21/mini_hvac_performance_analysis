import streamlit as st
from apps import app_mini_hvac, app_plant_hvac  # assume you move your app code into an `apps` folder

# Must be first Streamlit command
st.set_page_config(page_title="HVAC Analysis Launcher", layout="wide")

# Page selection
option = st.selectbox("Choose Analysis:", ["Mini-HVAC Analysis", "Plant-HVAC Analysis"])

# Run selected app
if option == "Mini-HVAC Analysis":
    app_mini_hvac.run()  # define a `run()` function in mini_hvac.py
elif option == "Plant-HVAC Analysis":
    app_plant_hvac.run()  # define a `run()` function in plant_hvac.py
