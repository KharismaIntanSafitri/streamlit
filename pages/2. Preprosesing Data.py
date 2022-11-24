import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib
from io import StringIO, BytesIO
import urllib.request
import joblib
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os,sys
from scipy import stats

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
    # uplod file
    data = pd.read_csv(uploaded_file)
    st.write(" **Nama File Anda :** ", uploaded_file.name)
   
    # view dataset asli
    st.header("Dataset Asli")
    X = data.drop(columns=["Type"])
    st.dataframe(X)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

    # view dataset NORMALISASI
    st.header("Dataset Normalisasi dan Transformasi")
    #  Tahap Normalisasi data sting ke kategori
    X = pd.DataFrame(X)
    X['Color'] = X['Color'].astype('category')
    X['Spectral_Class'] = X['Spectral_Class'].astype('category')
    cat_columns = X.select_dtypes(['category']).columns
    X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.dataframe( scaled_features)
    row, col = data.shape 
    st.caption(f"({row} rows, {col} cols)")

  