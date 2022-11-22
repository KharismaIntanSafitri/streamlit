import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.svm import SVC

from streamlit_option_menu import option_menu

st.title("Data Mining")
st.write("By Ahmad Rosyihuddin | 200411100126")
side_bar = st.sidebar.selectbox(
  
  'Pilih Metode',
  ('K-Nierneighboar', 'K-Means', 'Naive Bayes', 'Decission Tree')
)

selected = option_menu(
    menu_title  = "Metode Data Mining",
    options     = ["View Data","Preprocessing","Modeling","Implementation","Testing"],
    icons       = ["data","Process","model","implemen","Test"],
    orientation = "horizontal",
)


if selected == "View Data":
    st.write("""# Import Data""")
    uploaded_files = st.file_uploader("Upload file CSV", accept_multiple_files=True)
    for uploaded_file in uploaded_files:
        global data
        data = pd.read_csv(uploaded_file)
        st.write("Nama File Anda = ", uploaded_file.name)
        st.dataframe(data)
        st.write("Menampilkan 5 data teratas")
        st.dataframe(data.head())
        st.write("Melihat tipe-tipe data")
        st.dataframe(data.dtypes)

elif selected == "Preprocessing":
    st.write("# Preprocessing")
    # data.head()


elif selected == "Modeling":
    st.write("# Modeling")
    # data.head()

elif selected == "Modeling":
    st.write("# Modeling")
    # data.head()

elif selected == "Implementation":
    st.write("# Implementation")

else:
    st.write("# Testing")
    st.number_input('Masukkan fitur')