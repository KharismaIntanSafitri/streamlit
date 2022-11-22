import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_extras.app_logo import add_logo
import matplotlib.pyplot as plt

# add_logo("http://placekitten.com/150/150")
#st.markdown("# Decission Tree")
# selected = option_menu(
#     menu_title  = "Proses Metode Decission Tree",
#     options     = ["View Data","Preprocessing","Modeling","Implementation","Testing"],
#     icons       = ["data","Process","model","implemen","Test"],
#     orientation = "horizontal",
# )
# dataset = pd.read_csv("Dataset/BitcoinDataset.csv")
# if selected == "View Data":
#   st.write(dataset)

# elif selected == "Preprocessing":
#   data = dataset[['Close']].set_index(dataset['Date'])
#   data.plot(figsize=(21,10)).autoscale(axis='x',tight=True)