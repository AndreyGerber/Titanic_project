import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("train.csv")

st.title("Titanic : binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling"]
page=st.sidebar.radio("Go to", pages)

if page == pages[0] : 
  st.write("### Presentation of data")

st.dataframe(df.head(10))
st.write(df.shape)
st.dataframe(df.describe())

if st.checkbox("Show NA") :
  st.dataframe(df.isna().sum())


if page == pages[1] : 
  st.write("### DataVizualization")

fig = plt.figure()
sns.countplot(x = 'Survived', data = df)
st.pyplot(fig)