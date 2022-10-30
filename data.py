import streamlit as st

# set up libraries
import numpy as np
import pandas as pd

# get data and cache it
@st.cache # cache our data so the app doesn't constantly reload
def get_data(filename):
    basket_data = pd.read_excel(filename)
    return basket_data

df = get_data('Global Superstore.xls').sort_values('Order Date', ascending = True)
df['Order year'] = df['Order Date'].dt.year
df['Order period'] = df['Order Date'].dt.day
df['Ship period'] = df['Ship Date'].dt.day
