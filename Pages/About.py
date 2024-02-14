import streamlit as st
import pandas as pd

DataSetLink = "https://www.kaggle.com/competitions/dog-breed-identification/data"

st.set_page_config(
    page_title="Tokkyman",
    page_icon="🐶",
)

df = pd.read_csv("Csv/labels.csv")
st.title("About this project")

st.write("""This project purpose on applied deep learning with interactive web application, 
         Do not use this project on any production!
        """)

st.title("How was it made?")

st.write("""The original data set coming from [kaggle](%s), there provide 120 of dog breeds with 10,222 of total 
         image(training data set) and 10,357 image (test data set)
        """% DataSetLink)

st.write("""I use tranfer learning method with mobilenetV2(035-224-classification), 
        There are no evaluation on test data set since the original data set doesn't provie the true label,
        The evaluation on training data set show 99% of accuracy         
        """)

st.write("""I will rebuild the model again with same data set but split the data into 70:10:20 
         to show the test evaluation and try to find the best tune hyperparameter, 
         I will put the link to the new project if it done      
        """)

st.header("Check the dog breeds")
st.write("Here are the list of breeds of dog that use to train model")
st.dataframe(df["breed"].unique())
