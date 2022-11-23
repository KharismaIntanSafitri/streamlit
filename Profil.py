import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from sklearn.utils.validation import joblib

# intial template
px.defaults.template = "plotly_dark"
px.defaults.color_continuous_scale = "reds"

# create content
st.markdown("#Profil")
st.title("Profil Mahasiswa")
st.container()
st.write(''' 
            Hai, perkenalkan nama saya Kharisma Intan, atau biasa dipanggil Kharisma dengan nomor induk mahasiswa 200411100010. Saat ini saya sedang mengerjakan project matakuliah datamining. 
            \n**Apa itu Datamining?**
            \n**Datamining** adalah suatu proses pengumpulan informasi dan data yang penting dalam jumlah yang besar atau big data. 
            \nKali ini saya akan mencoba melakukan penambangan data dan klasifikasi terhadap data jenis jenis bintang. Untuk pertnyaan lebih lanjut silahkan bertanya melalui email saya
            \nEmail : kharismaintan2001@gmail.com 
            \nGithub : KharismaIntanSafitri 
            \nInstagram : kharisma_khar 
            \nYuk, coba dan belajar bersama di website Klasifikasi bintang ini. Sampai jumpaa and BYE !
        ''')





