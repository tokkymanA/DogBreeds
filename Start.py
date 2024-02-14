import streamlit as st
from st_pages import Page, add_page_title, show_pages
from PIL import Image
import io
from iden import Iden

show_pages(
    [
        Page("Start.py", "Home", "🏠"),
        Page("Pages/About.py", "Project Detailed", "📑")
    ]
)

st.set_page_config(
    page_title="Tokkyman",
    page_icon="🐶",
)

st.header("Dog Breed Identification Project🐕‍🦺")
st.subheader("Check the dog breed with just a sec!")
st.image("Picture/TestDog.png", caption="Test result example")

with st.container():
    st.subheader("Upload your lovely dog picture to identify the breed!")
    st.write("""Have you ever wondered what breed your dog is?, if yes you come for the right place!
    """)
    st.write("This machine learning project trained 120 dog breeds covering popular breeds like Labrador Retrievers, Golden Retrievers, Shih Tzu and more")
    st.write("Just upload your lovely dog picture and wait the result")


with st.form("UploadForms"):

    uploaded = st.file_uploader(label="Upload your lovely dog picture here!", type=['jpg'])
    submitted = st.form_submit_button("Confirm")

if submitted and uploaded:
   
    bytes_data = uploaded.getvalue()
    try:
        with st.spinner('Load model...'):
            st.image(image = Image.open(io.BytesIO(bytes_data)))
            ModelLoad = Iden(bytes_data) 
            result = ModelLoad.result
            st.header(f"{result[0]}")
            st.write(f"It's seem like your dog breed is {result[0]} with {result[1] * 100:.2f}% of probability!")
            st.image(image = Image.open(io.BytesIO(bytes_data)))
        print("Load model successful...")

    except Exception as error :
        st.warning("Can't load model", icon="⚠️")
        st.write(error)
        print(error)
        print("Load model fail")




