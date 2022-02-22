import requests
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf
from PIL import Image, ImageOps
# from tensorflow.keras.preprocessing import image
from img_classification import teachable_machine_classification

# st.set_page_config(layout="wide")

st.title("Applied AI Project - Demo")


# logo_applied_roots = Image.open("./data/webapp_logos/Applied_Roots_logo.png")
# logo_uoh = Image.open("./data/webapp_logos/University_of_Hyderabad_Logo.png")

st.set_option('deprecation.showfileUploaderEncoding', False)
st.sidebar.title("About")
st.sidebar.info(
    " This project is done as a part of the Post Graduate Diploma \
                thesis or end semester project. \
                This short demo helps us to pick a image and get the prediction of the developed Multi class classifer "
)
st.sidebar.info("Used Dataset- CDiscount Classification Challenge.")
# st.sidebar.image(logo_applied_roots, width=300)  # , use_column_width = 'auto')
# st.sidebar.image(logo_uoh, width=300, use_column_width="auto")

st.write("Please select any  one of the methods below - Upload Image or Take a Picture")
# image = st.file_uploader("Choose an Image")


# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = tf.keras.models.load_model(
#         'D:\PYTHON PROJECTS\Cdiscount\model\Cdiscount.hdf5')
#     return model


# with st.spinner('Model is being loaded...'):
#     model = load_model()

st.write("""
# Cdiscount Image Classification
""")

file = st.file_uploader(
    "Please upload any ecommerce Image", type=["jpg", "png"])

if file is None:
    st.text("Please upload an Image File")

else:

    img = Image.open(file)
    st.image(img, use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_label, score,  percent_confidence = teachable_machine_classification(
        img)
    st.write("""# Metrics""")
    # st.write(predicted_label)
    # st.write(np.max(score)*100)
    col1, col2 = st.columns(2)
    col1.metric('Predicted Label', predicted_label)
    col2.metric('Confidence Score %', np.round(percent_confidence, 3))
    # col3.metric('English Translation', en_translate)
