import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
# from googletrans import Translator


# Create dictionaries for quick lookup of category_id to category_idx mapping.
def make_category_tables():
    categories_df = pd.read_csv("categories.csv", index_col=0)

    cat2idx = {}
    idx2cat = {}
    i = 0
    for ir in categories_df.itertuples():

        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id

    return cat2idx, idx2cat


def teachable_machine_classification(img):
    # Load the model
    model = tf.keras.models.load_model(
        'D:\PYTHON PROJECTS\Cdiscount\Cdiscount.hdf5')
    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 180, 180, 3), dtype=np.float32)
    image = img
    # Image Sizing
    size = (180, 180)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn image into a numpy array
    image_array = np.asarray(image)

    # Load the image into the array
    data[0] = image_array

    # run the infrence
    prediction = model.predict(data)
    index = np.argmax(prediction)

    # Importing categiroy table

    cat2idx, idx2cat = make_category_tables()
    categories_df = pd.read_csv("categories.csv", index_col=0)
    cat = categories_df.reset_index()
    predicted_label = cat.loc[cat['category_id'] == idx2cat[index]].to_dict()
    category_l3 = predicted_label["category_level3"][index]
    # tranlating french Word
    # translator = Translator()
    score = tf.nn.softmax(prediction[0])
    percent_confidence = np.max(score)*100
    # return position of the highest probability
    return category_l3, score,  percent_confidence
