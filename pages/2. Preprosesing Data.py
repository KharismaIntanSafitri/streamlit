import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.markdown("#2. Preprosesing Data")


# st.header("Input Data Sample")
# datain = st.text_input('Masukkan dataset', '')
st.title("Preprosesing Data")
st.write("Preprocessing adalah proses menyiapkan data dasar atau inti sebelum melakukan proses lainnya. \nPada dasarnya data preprocessing dapat dilakukan dengan membuang data yang tidak sesuai atau mengubah data menjadi bentuk yang lebih mudah untuk diproses oleh sistem. \nProses pembersihan meliputi penghilangan duplikasi data, pengisian atau penghapusan data yang hilang, pembetulan data yang tidak konsisten, dan pembetulan salah ketik. \nSeperti namanya, normalisasi dapat diartikan secara sederhana sebagai proses menormalkan data dari hal-hal yang tidak sesuai.")
st.header("Import Data")
uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    data = pd.read_csv(uploaded_file)
    st.write(" **Nama File Anda :** ", uploaded_file.name)
    st.header("Dataset Asli")
    st.dataframe(data)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

    st.header("Dataset Normalisasi dan Transformasi")
    st.dataframe(data)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")