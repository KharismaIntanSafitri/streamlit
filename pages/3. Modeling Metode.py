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

st.markdown("#3. Modeling Data")

    # ambil data
url = "https://raw.githubusercontent.com/KharismaIntanSafitri/datamining/main/data_bintang_acak.csv"

header  = ['Temperatur', 'Luminious', 'Radius', 'Magnitudo Absolute', 'Color',"Spectral Type", "Type" ]
data = pd.read_csv(url, names = header)
data = data.drop(0, axis=0)
X = data.drop(columns=['Type'])
X = pd.DataFrame(X)
X['Color'] = X['Color'].astype('category')
X["Spectral Type"] = X["Spectral Type"].astype('category')
cat_columns = X.select_dtypes(['category']).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.cat.codes)

scaler = MinMaxScaler()
cut = scaler.fit_transform(X.drop(columns=["Color", "Spectral Type"]))
scaled = np.column_stack([cut, X[["Color", "Spectral Type"]]])
features_names = X.columns.copy()
scaled_features = pd.DataFrame(scaled, columns=features_names)

import joblib
filename = "normalisasi_bintang.sav"
joblib.dump(scaler, filename) 

y = data['Type'].values

# encoder label
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y_baru = le.fit_transform(y)

# split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y_baru, test_size=0.2, random_state=1)

# inisialisasi knn
my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
knn.fit(X_train, y_train)

pred_test = knn.predict(X_test)

vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour : {accuracy_score(y_test, pred_test) * 100 :.2f} %'

filenameModelKnnNorm = 'modelKnnNorm.pkl'
joblib.dump(knn, filenameModelKnnNorm)

# inisialisasi model gausian
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

filenameModelGau = 'modelGau.pkl'
joblib.dump(gnb, filenameModelGau)

y_pred = gnb.predict(X_test)

vg = f'Hasil akurasi dari pemodelan Gausian : {accuracy_score(y_test, y_pred) * 100 :.2f} %'

# inisialisasi model decision tree
from sklearn.tree import DecisionTreeClassifier
d3 = DecisionTreeClassifier()
d3.fit(X_train, y_train)

filenameModeld3 = 'modeld3.pkl'
joblib.dump(d3, filenameModeld3)

y_pred = d3.predict(X_test)

vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} %'

# inisialisasi model k-mean
from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(
base_estimator = gnb,
n_estimators = 300,
max_samples = 0.9,
oob_score = True
)
bagging.fit(X_train, y_train)

filenameModelBagg = 'modelBagg.pkl'
joblib.dump(bagging, filenameModelBagg)

y_pred = bagging.predict(X_test)
vbag = f'Hasil akurasi dari pemodelan bagging : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
# tampilan tabs
K_Nearest_Naighbour, Gausian, Decision_Tree, Bagging = st.tabs(["K-Nearest aighbour", "Naive Bayes Gausian", "Decision Tree", "Bagging"])


with K_Nearest_Naighbour:
    st.header("K-Nearest Neighbour")
    st.write("Algoritma KNN mengasumsikan bahwa sesuatu yang mirip akan ada dalam jarak yang berdekatan atau bertetangga. Artinya data-data yang cenderung serupa akan dekat satu sama lain. KNN menggunakan semua data yang tersedia dan mengklasifikasikan data atau kasus baru berdasarkan ukuran kesamaan atau fungsi jarak. Data baru kemudian ditugaskan ke kelas tempat sebagian besar data tetangga berada.")
    st.header("Pengkodean")
    st.text("""
    my_param_grid = {'n_neighbors':[2,3,5,7], 'weights': ['distance', 'uniform']}
    GridSearchCV(estimator=KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn = GridSearchCV(KNeighborsClassifier(), param_grid=my_param_grid, refit=True, verbose=3, cv=3)
    knn.fit(X_train, y_train)

    pred_test = knn.predict(X_test)

    vknn = f'Hasil akurasi dari pemodelan K-Nearest Neighbour : {accuracy_score(y_test, pred_test) * 100 :.2f} %'
    """)
    st.header("Hasil Akurasi")
    st.write(vknn)
    
    

with Gausian:
    st.header("Naive Bayes Gausian")
    st.write("Metode yang juga dikenal sebagai Naive Bayes Classifier ini menerapkan teknik supervised klasifikasi objek di masa depan dengan menetapkan label kelas ke instance/catatan menggunakan probabilitas bersyarat. \nProbabilitas bersyarat adalah ukuran peluang suatu peristiwa yang terjadi berdasarkan peristiwa lain yang telah (dengan asumsi, praduga, pernyataan, atau terbukti) terjadi \nRumus: P(A???B) = P(B???A)P(A)P(B). Adapun salah satu jenis naive bayes adalah gausian. Distribusi Gaussian adalah asumsi pendistribusian nilai kontinu yang terkait dengan setiap fitur berisi nilai numerik. Ketika diplot, akan muncul kurva berbentuk lonceng yang simetris tentang rata-rata nilai fitur.")
    st.header("Pengkodean")
    st.text("""
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    filenameModelGau = 'modelGau.pkl'
    joblib.dump(gnb, filenameModelGau)

    y_pred = gnb.predict(X_test)

    vg = f'Hasil akurasi dari pemodelan Gausian : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
        """)
    st.header("Hasil Akurasi")
    st.write(vg)   

with Decision_Tree:
    st.header("Decision Tree")
    st.write("Konsep Decision tree  adalah dengan cara menyajikan algoritma dengan pernyataan bersyarat, \nyang meliputi cabang untuk mewakili langkah-langkah pengambilan keputusan yang dapat mengarah pada hasil yang menguntungkan.")
    st.header("Pengkodean")
    st.text(""" 
    from sklearn.tree import DecisionTreeClassifier
    d3 = DecisionTreeClassifier()
    d3.fit(X_train, y_train)

    filenameModeld3 = 'modeld3.pkl'
    joblib.dump(d3, filenameModeld3)

    y_pred = d3.predict(X_test)

    vd3 = f'Hasil akurasi dari pemodelan decision tree : {accuracy_score(y_test, y_pred) * 100 :.2f} %'
    """)
    st.header("Hasil Akurasi")
    st.write(vd3)   

with Bagging:
    st.header("Bagging")
    st.write("Bagging merupakan metode yang dapat memperbaiki hasil dari algoritma klasifikasi machine learning dengan menggabungkan klasifikasi prediksi dari beberapa model. Hal ini digunakan untuk mengatasi ketidakstabilan pada model yang kompleks dengan kumpulan data yang relatif kecil.")
    st.header("Pengkodean")
    st.text(""" 
    bagging = BaggingClassifier(
    base_estimator = gnb,
    n_estimators = 300,
    max_samples = 0.9,
    oob_score = True
    )
    bagging.fit(X_train, y_train)

    filenameModelBagg = 'modelBagg.pkl'
    joblib.dump(bagging, filenameModelBagg)

    y_pred = bagging.predict(X_test)

    vbag = f'acuracy = {accuracy_score(y_test, y_pred) * 100 :.2f}%'
    """)
    st.header("Hasil Akurasi")
    st.write(vbag)