from datetime import datetime
import altair as alt
import joblib
import pandas as pd
import pytz
import streamlit as st
import os


df = pd.read_csv('data.csv')


@st.cache(persist = True)
def get_total_vaccinations():
    pass


@st.cache(persist = True)
def get_lines(data):
    pass


if __name__ == "__main__":
    #df = read_data()

