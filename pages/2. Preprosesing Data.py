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
st.write("""
Preprocessing adalah proses menyiapkan data dasar atau inti sebelum melakukan proses lainnya. 
\nPada dasarnya data preprocessing dapat dilakukan dengan membuang data yang tidak sesuai atau mengubah data menjadi bentuk yang lebih mudah untuk diproses oleh sistem. 
\nProses pembersihan meliputi penghilangan duplikasi data, pengisian atau penghapusan data yang hilang, pembetulan data yang tidak konsisten, dan pembetulan salah ketik. 
\nSeperti namanya, normalisasi dapat diartikan secara sederhana sebagai proses menormalkan data dari hal-hal yang tidak sesuai. 
\nDalam proses klasifikasi bintang kali ini tahap yang dilakukan sebgaai berikut :
\n1. Memisahkan data fitur dengan data target pada kolom Type, dengan cara mendrop kolom Type
\n2. Merubah data kategorial dari kolom Color dan Spectral Type menjadi data numerik agar dapat dihitung
\n3. Melakukan normalisasi data dengan menggunakan metode min-max scaler pada data fitur yang bertipe numerik untuk membuat beberapa variabel memiliki rentang nilai yang sama sehingga analisa statistik lebih mudah
""")
st.header("Import Data")
st.write("Menggunakan Link Dataset Berikut :")
st.write("https://raw.githubusercontent.com/KharismaIntanSafitri/datamining/main/data_bintang_acak.csv")
uploaded_files  = "https://raw.githubusercontent.com/KharismaIntanSafitri/datamining/main/data_bintang_acak.csv"

    # uplod file
header  = ['Temperatur', 'Luminious', 'Radius', 'Magnitudo Absolute', 'Color',"Spectral Type", "Type" ]
data = pd.read_csv(uploaded_files, names=header )
   
    # view dataset asli
st.header("Dataset Asli")
    # X = data.drop(columns=["Type"])
X = data.drop(0, axis=0)
st.dataframe(X)
row, col = data.shape 
st.caption(f"({row} rows, {col} cols)")

    # view dataset NORMALISASI
st.header("Dataset Hasil Preprocessing")
    #  Tahap Normalisasi data sting ke kategori
X = pd.DataFrame(X)
X['Color'] = X['Color'].astype('category')
X["Spectral Type"] = X["Spectral Type"].astype('category')
cat_columns = X.select_dtypes(['category']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)
    
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
cut = scaler.fit_transform(X.drop(columns=["Color", "Spectral Type"]))
scaled = np.column_stack([cut, X[["Color", "Spectral Type"]]])
features_names = X.columns.copy()
scaled_features = pd.DataFrame(scaled, columns=features_names)
scaled_features = scaled_features.drop(columns=['Type'])

st.dataframe( scaled_features)
row, col = data.shape 
st.caption(f"({row} rows, {col} cols)")
    

  