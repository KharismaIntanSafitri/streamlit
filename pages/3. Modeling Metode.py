import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

st.markdown("#3. Modeling Data")

K_Nearest_Naighbour, Naive_Bayes, Decision_Tree, K_Mean = st.tabs(["K-Nearest aighbour", "Naive Bayes", "Decision Tree", "K-Mean"])

with K_Nearest_Naighbour:
    st.header("K-Nearest Neighbour")
    st.write("Algoritma KNN mengasumsikan bahwa sesuatu yang mirip akan ada dalam jarak yang berdekatan atau bertetangga. Artinya data-data yang cenderung serupa akan dekat satu sama lain. /nKNN menggunakan semua data yang tersedia dan mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan atau fungsi jarak. Data baru kemudian ditugaskan ke kelas tempat sebagian besar data tetangga berada.")
    st.header("Pengkodean")
    st.text("Hello world")
    st.header("Hasil Akurasi")
    st.write("Hasil akurasi dari pemodelan K-Nearest Neighbour : 30%")
    
    

with Naive_Bayes:
    st.header("Naive Bayes")
    st.write("Metode yang juga dikenal sebagai Naive Bayes Classifier ini menerapkan teknik supervised klasifikasi objek di masa depan dengan menetapkan label kelas ke instance/catatan menggunakan probabilitas bersyarat. \nProbabilitas bersyarat adalah ukuran peluang suatu peristiwa yang terjadi berdasarkan peristiwa lain yang telah (dengan asumsi, praduga, pernyataan, atau terbukti) terjadi \nRumus: P(A│B) = P(B│A)P(A)P(B)")
    st.header("Pengkodean")
    st.text("Hello world")
    st.header("Hasil Akurasi")
    st.write("Hasil akurasi dari pemodelan Naive Bayes : 48%")   

with Decision_Tree:
    st.header("Decision Tree")
    st.write("Konsep Decision tree  adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, \nyang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan.")
    st.header("Pengkodean")
    st.text("Hello world")
    st.header("Hasil Akurasi")
    st.write("Hasil akurasi dari pemodelan Decision Tree: 48%")   

with K_Mean:
    st.header("K-Means")
    st.write("Prinsip utama K-Means adalah menyusun k prototype atau pusat massa (centroid) dari sekumpulan data berdimensi. \nSebelum diterapkan proses algoritma K-means, dokumen akan di preprocessing terlebih dahulu. Kemudian dokumen direpresentasikan sebagai vektor yang memiliki term dengan nilai tertentu.")
    st.header("Pengkodean")
    st.text("Hello world")
    st.header("Hasil Akurasi")
    st.write("Hasil akurasi dari pemodelan K-Means: 48%")