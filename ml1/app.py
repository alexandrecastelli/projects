# python -m venv .venv
# source .venv/bin/activate
# deactivate
# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
import os

st.set_page_config(
    page_title="Master League 1",
    page_icon=":racing_car:",
    menu_items={'Get Help': 'mailto:john@example.com',
                'Report a bug': 'mailto:john@example.com',
                'About': 'text'}
)

st.logo("/workspaces/projects/ml1/images/ml1.png", size='large')

ml1 = pd.read_excel("/workspaces/projects/ml1/ml1.xlsx")


add_selectbox = st.sidebar.selectbox(
    'Contato', 
    ('Email', 'Home phone', 'Mobile phone')
    )


add_slider = st.sidebar.slider(
    "Seleciona um valor", 
    0.0, 100.0, (25.0, 75.0)
    )


st.write("# Welcome to ML1! 👋")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)