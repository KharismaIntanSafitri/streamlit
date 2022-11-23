import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.markdown("#4. Implementasi Model")


# st.header("Input Data Sample")
# datain = st.text_input('Masukkan dataset', '')
st.title("Implementasi Model")
st.write("Sebagai bahan eksperimen silahkan inputkan beberapa data yang akan digunakan sebagai data testing untuk pengklasifikasian")

st.header("Input Data Testing")
# create input
temperatur = st.number_input("Temperatur", 0.0, 50.0, step=0.1)
L = st.number_input("Luminocity", 0.0, 50.0, step=0.1)
R = st.number_input("Radius", 0.0, 50.0, step=0.1)
AM = st.number_input("Absolute Magnitudo", 0.0, 50.0, step=0.1)
color = st.text_input("Color", "General Color of Spectrum", )
spectral = st.text_input('Type of Spectral',"O,B,A,F,G,K,M" )

def submit():
    # input
    inputs = np.array([[temperatur, L, R, AM, color, spectral]])
    st.header("Data Input")
    st.write("Berikut ini tabel hasil input data testing yang akan diklasifikasi:")
    st.dataframe(inputs)

    st.header("Hasil Prediksi")
    K_Nearest_Naighbour, Naive_Bayes, Decision_Tree, K_Mean = st.tabs(["K-Nearest aighbour", "Naive Bayes", "Decision Tree", "K-Mean"])
    with K_Nearest_Naighbour:
        st.subheader("Model K-Nearest Neighbour")
        st.write("Hasil Klasifikaisi : Red Dwarf")

    with Naive_Bayes:
        st.subheader("Model Naive Bayes")
        st.write("Hasil Klasifikaisi : Brown Dwarf")

    with Decision_Tree:
        st.subheader("Model Decision Tree")
        st.write("Hasil Klasifikaisi : Red Dwarf ")

    with K_Mean:
        st.subheader("Model K-Means")
        st.write("Hasil Klasifikaisi : Red Dwarf ")

submitted = st.button("Submit")
if submitted:
    submit()





