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

st.markdown("#4. Implementasi Model")


st.title("Implementasi Model")
st.write("Sebagai bahan eksperimen silahkan inputkan beberapa data yang akan digunakan sebagai data testing untuk pengklasifikasian")

st.header("Input Data Testing")
# create input
t = st.number_input("Temperatur")
l = st.number_input("Luminocity")
r = st.number_input("Radius")
ma = st.number_input("Absolute Magnitudo")
clr = st.selectbox(
    'Color',
    ('Blue', 'Blue White', 'Orange', 'Orang-Red', 'Pale yellow orange', 'Red', 'White', 'White-Yellow', 'Whitish', 'Yellowish', 'Yellowish White', 'yellow-White' ))
sp = st.selectbox(
    'Magnitudo Absolute',
    ('A', 'B', 'F', 'G', 'K', 'M', 'O'))

def submit():
    # input
    header  = ['Temperatur', 'Luminious', 'Radius', 'Magnitudo Absolute', 'Color',"Spectral Type", "Type" ]
    url = "https://raw.githubusercontent.com/KharismaIntanSafitri/datamining/main/data_bintang_acak.csv" 
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
    # olah Inputan
    a = np.array([[t, l, r, ma, clr, sp ]])

    test_data = np.array(a).reshape(1, -1)
    test_data = pd.DataFrame(test_data, columns =['Temperatur', 'Luminious', 'Radius', 'Magnitudo Absolute', 'Color','Spectral Type' ])

    test_data = pd.DataFrame(test_data)
    test_data['Color'] = test_data['Color'].astype('category')
    test_data['Spectral Type'] = test_data['Spectral Type'].astype('category')
    cat_columns = test_data.select_dtypes(['category']).columns
    test_data[cat_columns] = test_data[cat_columns].apply(lambda x: x.cat.codes)

    scaler = joblib.load(filename)
    test_d = scaler.fit_transform(test_data)
    # pd.DataFrame(test_d)

    # load knnv
    knn = joblib.load(filenameModelKnnNorm)
    pred = knn.predict(test_d)

    # load gausian
    gnb = joblib.load(filenameModelGau)
    pred = gnb.predict(test_d)

    # load gdecision tree
    d3 = joblib.load(filenameModeld3)
    pred = d3.predict(test_d)

    # load kmean
    bagging = joblib.load(filenameModelBagg)
    pred = bagging.predict(test_d)





    # button
    st.header("Data Input")
    st.write("Berikut ini tabel hasil input data testing yang akan diklasifikasi:")
    st.dataframe(a)
    
    st.header("Hasil Prediksi")
    K_Nearest_Naighbour, Naive_Bayes, Decision_Tree, Bagging = st.tabs(["K-Nearest aighbour", "Naive Bayes Gausian", "Decision Tree", "Bagging"])
    
    with K_Nearest_Naighbour:
        st.subheader("Model K-Nearest Neighbour")
        pred = knn.predict(test_d)
        if pred[0]== 0:
            st.write("Hasil Klasifikaisi : Red Dwarf")
        elif pred[0]== 1 :
            st.write("Hasil Klasifikaisi : Brown Dwarf")
        elif pred[0]== 2:
            st.write("Hasil Klasifikaisi : White Dwarf")
        elif pred[0]== 3:
            st.write("Hasil Klasifikaisi : Main Sequence")
        elif pred[0]== 4:
            st.write("Hasil Klasifikaisi : Super Giants ")
        elif pred[0]== 5:
            st.write("Hasil Klasifikaisi : Hyper Giants")
        else:
            st.write("Hasil Klasifikaisi : New Category")
        

    with Naive_Bayes:
        st.subheader("Model Naive Bayes Gausian")
        pred = gnb.predict(test_d)
        if pred[0]== 0:
            st.write("Hasil Klasifikaisi : Red Dwarf")
        elif pred[0]== 1 :
            st.write("Hasil Klasifikaisi : Brown Dwarf")
        elif pred[0]== 2:
            st.write("Hasil Klasifikaisi : White Dwarf")
        elif pred[0]== 3:
            st.write("Hasil Klasifikaisi : Main Sequence")
        elif pred[0]== 4:
            st.write("Hasil Klasifikaisi : Super Giants ")
        elif pred[0]== 5:
            st.write("Hasil Klasifikaisi : Hyper Giants")
        else:
            st.write("Hasil Klasifikaisi : New Category")

    with Decision_Tree:
        st.subheader("Model Decision Tree")
        pred = d3.predict(test_d)
        if pred[0]== 0:
            st.write("Hasil Klasifikaisi : Red Dwarf")
        elif pred[0]== 1 :
            st.write("Hasil Klasifikaisi : Brown Dwarf")
        elif pred[0]== 2:
            st.write("Hasil Klasifikaisi : White Dwarf")
        elif pred[0]== 3:
            st.write("Hasil Klasifikaisi : Main Sequence")
        elif pred[0]== 4:
            st.write("Hasil Klasifikaisi : Super Giants ")
        elif pred[0]== 5:
            st.write("Hasil Klasifikaisi : Hyper Giants")
        else:
            st.write("Hasil Klasifikaisi : New Category")

    with Bagging:
        st.subheader("Model Bagging")
        pred = bagging.predict(test_d)
        if pred[0]== 0:
            st.write("Hasil Klasifikaisi : Red Dwarf")
        elif pred[0]== 1 :
            st.write("Hasil Klasifikaisi : Brown Dwarf")
        elif pred[0]== 2:
            st.write("Hasil Klasifikaisi : White Dwarf")
        elif pred[0]== 3:
            st.write("Hasil Klasifikaisi : Main Sequence")
        elif pred[0]== 4:
            st.write("Hasil Klasifikaisi : Super Giants ")
        elif pred[0]== 5:
            st.write("Hasil Klasifikaisi : Hyper Giants")
        else:
            st.write("Hasil Klasifikaisi : New Category")

submitted = st.button("Submit")
if submitted:
    submit()





