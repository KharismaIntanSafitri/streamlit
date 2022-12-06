import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.markdown("#1. Informasi Data")
# create content
st.title("Classification of Stars")
st.container()
st.write("Website ini bertujuan untuk mengklasifikasikan jenis bintang berdsarkan temperatur, luminious, magnitudo absolute, warna dan tipe spectral. \nBerdasarkan informasi dari Wikipedia tipe spectral bintang  dibagi menjadi Tujuh yakni O,B,A,F,G dan M. \nTarget pengelempokan jenis bintang ini dibagi menjadi lima yaitu Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , Super Giants, Hyper Giants ")

st.header("Informasi Data")

# read data
st.write("""
Dalam dataset yang digunakan terdapat beberapa variabel untuk pengklasifikasian bintang:
* Temperatur          : Satuan Kelvin (K)
* Luminious           : Luminosity  / Lo -> ( Lo = 3.828 x 10^26 Watts)
* Radius              : Radius / Ro -> ( Ro = 6.9551 x 10^8 m)
* Magnitudo Absolute  : Magnitudo absolute dalam satuan Mv
* Color               : General Color of Spectrum
* Spectral Type       : O,B,A,F,G,K,M

Adapun target Pengklasifikasian dengan pengkodean 0-5 dapat dilihat sebagai berikut:
* Red Dwarf - 0
* Brown Dwarf - 1
* White Dwarf - 2
* Main Sequence - 3
* Super Giants - 4
* Hyper Giants - 5
""")

st.header("Sumber Data")
st.write("Dataset yang digunakan dalam percobaan ini diambil dari Kagle dengan jumlah data sebanyak 241 data dengan 6 fitur dengan target pengelompokan menjadi 5 jeis bintang")
st.caption('link datasets : https://github.com/KharismaIntanSafitri/datamining/blob/main/data%20bintang%20-%20Sheet1.csv')
data = "https://raw.githubusercontent.com/KharismaIntanSafitri/datamining/main/data_bintang_acak.csv"
header  = ['Temperatur', 'Luminious', 'Radius', 'Magnitudo Absolute', 'Color',"Spectral Type ", "Type" ]
df = pd.read_csv(data, names=header )
df = df.head()
hapus = df.drop(0, axis=0)
st.dataframe(hapus)
