import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

# create content
st.title("Classification of Stars")
st.container()
st.write("Website ini bertujuan untuk mengklasifikasikan jenis bintang berdsarkan temperatur, luminious, massa, warna dan tipe spectral. \nBerdasarkan informasi dari Wikipedia tipe spectral bintang  dibagi menjadi Tujuh yakni O,B,A,F,G dan M. \nTarget pengelempokan jenis bintang ini dibagi menjadi lima yaitu Red Dwarf, Brown Dwarf, White Dwarf, Main Sequence , Super Giants, Hyper Giants ")

st.header("Input Data Sample")
datain = st.text_input('Masukkan dataset', '')

st.header("Informasi Data Sample")

st.markdown("#1 informasi")




# read data
df = pd.read_csv("https://raw.githubusercontent.com/KharismaIntanSafitri/datamining/main/data%20bintang%20-%20Sheet1.csv")
st.text("""
Keterangan Fitur Dataset:
* Temperatur          : Satuan Kelvin (K)
* L                   : Luminosity  / Lo -> ( Lo = 3.828 x 10^26 Watts)
* R                   : Radius / Ro -> ( Ro = 6.9551 x 10^8 m)
* A_M                 : Magnitudo absolute dalam satuan Mv
* Color               : General Color of Spectrum
* Spectral_Type       : O,B,A,F,G,K,M

Target Pengklasifikasian dengan pengkodean 0-5:
* Red Dwarf - 0
* Brown Dwarf - 1
* White Dwarf - 2
* Main Sequence - 3
* Super Giants - 4
* Hyper Giants - 5
""")
st.caption('link datasets : https://github.com/KharismaIntanSafitri/datamining/blob/main/data%20bintang%20-%20Sheet1.csv')
st.write(df)
row, col = df.shape
st.caption(f"({row} rows, {col} 