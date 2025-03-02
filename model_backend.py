# showing uploaded image preview

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# import tensorflow as tf
# 
# app = Flask(__name__)
# CORS(app)  # This enables CORS for all routes
# 
## Load the trained model
# model = load_model('models/imageclassifier.h5')
# 
# @app.route('/predict', methods=['POST'])
# def predict():
    # file = request.files['file']  # Receive the image from the frontend
    # img_path = f"./uploads/{file.filename}"
    # file.save(img_path)
    # print("File saved successfully...")
# 
##    Preprocess the image
    # img = cv2.imread(img_path)
    # img_resized = tf.image.resize(img, (256, 256))
    # img_resized = img_resized / 255.0
    # img_resized = np.expand_dims(img_resized, axis=0)
    # print("Model is ready to predict....")
# 
##    Make prediction
    # prediction = model.predict(img_resized)
    # print("Model predicted successfully...")
    # print("Predicted value:", prediction)
# 
##    Interpret the result
    # result = "Multiple Sclerosis" if prediction > 0.5 else "Healthy"
    # print("Prediction:", result)
    # return jsonify({'prediction': result})
# 
# if __name__ == '__main__':
    # app.run(debug=True)
# 

# Integrating with streamlit

# import os
# import subprocess

# # Install missing dependencies
# try:
#     import tensorflow
# except ModuleNotFoundError:
#     subprocess.run(["pip", "install", "tensorflow", "--no-cache-dir"])

# try:
#     import cv2
# except ModuleNotFoundError:
#     subprocess.run(["pip", "install", "opencv-python", "--no-cache-dir"])

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf

# Load the trained model
model = load_model('models/imageclassifier.h5')

st.title("Multiple Sclerosis MRI Prediction")
st.header("Upload an MRI Image")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    # st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)

    st.write("Classifying...")

    # Save the uploaded file temporarily
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Preprocess the image
    img_resized = tf.image.resize(img, (256, 256))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)

    # Make prediction
    prediction = model.predict(img_resized)
    result = "Multiple Sclerosis" if prediction > 0.5 else "Healthy"

    # st.write(f"Prediction: **{result}**")

    col1, col2 = st.columns(2)
    col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    col2.write(f"Prediction: **{result}**")

# Commands to run 
#  1111  python3 -m venv myenv
#  1112  source myenv/bin/activate
#  1113  pip install streamlit tensorflow numpy opencv-python
#  1115  streamlit run model_backend.py 
#  1116  deactivate
