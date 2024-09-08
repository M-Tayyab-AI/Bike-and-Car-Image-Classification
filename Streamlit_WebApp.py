import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

model = load_model(r'D:\Tayyab\Projects\Kaggle\Bike and car classification\Car-Bike-Dataset\Model\car_bike_classifier.h5')

st.set_page_config(page_title="Bike or Car Classifier", page_icon="🏍️/🚗", layout="wide")

st.title("🏍️ Bike or Car Classifier 🚗")
st.markdown("""
    Upload an image of a bike or a car, and this app will classify it for you!
""")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image in the main app
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert the file to an image and preprocess it
    img = Image.open(uploaded_file)
    img_resized = img.resize((150, 150))
    img_array = image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, axis=0)

    # Predict the class
    predictions = model.predict(img_array)
    class_names = ['Bike', 'Car']
    prediction_label = class_names[int(predictions[0][0] > 0.5)]

    # Display the prediction result
    st.markdown(f"### Prediction: **{prediction_label}**")
    st.markdown(f"**Classification:** {predictions[0][0]:.2f}")

    if prediction_label == 'Car':
        st.success("This is a car! 🚗")
    else:
        st.success("This is a bike! 🏍️")

st.markdown("""
    ---
    **Created by Tayyab**
""")
