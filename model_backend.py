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

# Integrating with streamlit
# ----------------------------------------------------
# import os
# os.system('pip uninstall -y tensorflow tensorflow-cpu && pip install tensorflow-cpu==2.9.1')

# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# import cv2
# import tensorflow as tf

# # Load the trained model
# model = load_model('models/imageclassifier.h5')

# st.title("Multiple Sclerosis MRI Prediction")
# st.header("Upload an MRI Image")

# # File uploader
# uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "bmp"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     # st.image(uploaded_file, caption="Uploaded MRI Image", use_column_width=True)
#     st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)

#     st.write("Classifying...")

#     # Save the uploaded file temporarily
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     img = cv2.imdecode(file_bytes, 1)

#     # Preprocess the image
#     img_resized = tf.image.resize(img, (256, 256))
#     img_resized = img_resized / 255.0
#     img_resized = np.expand_dims(img_resized, axis=0)

#     # Make prediction
#     prediction = model.predict(img_resized)
#     result = "Multiple Sclerosis" if prediction > 0.5 else "Healthy"

#     # st.write(f"Prediction: **{result}**")

#     col1, col2 = st.columns(2)
#     col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
#     col2.write(f"Prediction: **{result}**")

# ------------------------------------------------------------------------------------------------
# Commands to run 
#  1111  python3 -m venv myenv
#  1112  source myenv/bin/activate
#  1113  pip install streamlit tensorflow numpy opencv-python
#  1115  streamlit run model_backend.py 
#  1116  deactivate

# ------------------------------------------------------------------------------------------------
# Implementing grad-cam-visualization

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import tempfile

# Load the trained model
model = load_model('models/imageclassifier.h5')

# Ensure the model is built by calling it with a dummy input
# dummy_input = np.random.rand(1, 256, 256, 3).astype(np.float32)  # Adjust shape if needed
# model.predict(dummy_input)  # Forces model to initialize

dummy_input = tf.zeros((1, 256, 256, 3))  # Adjust to your input shape
_ = model(dummy_input)  # Run a dummy inference to initialize layers

st.title("Multiple Sclerosis MRI Prediction")
st.header("Upload an MRI Image")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "bmp"])

# def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
#     grad_model = tf.keras.models.Model(
#         # [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
#         [model.input], [model.get_layer(last_conv_layer_name).output, model.outputs[0]]
#     )
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         class_index = np.argmax(predictions[0])
#         loss = predictions[:, class_index]
#     grads = tape.gradient(loss, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#     conv_outputs = conv_outputs[0]
#     for i in range(pooled_grads.shape[-1]):
#         conv_outputs[:, :, i] *= pooled_grads[i]
#     heatmap = np.mean(conv_outputs, axis=-1)
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#     return heatmap

dummy_input = np.zeros((1, 256, 256, 3), dtype=np.float32)
_ = model.predict(dummy_input)  # Ensure model is built


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """Generates a Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        model.inputs,  # Use model.inputs instead of model.input
        [model.get_layer(last_conv_layer_name).output, model.output]
    )


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Assuming binary classification

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs *= pooled_grads
    heatmap = tf.reduce_mean(conv_outputs, axis=-1)

    heatmap = np.maximum(heatmap, 0)  # ReLU activation
    heatmap /= tf.reduce_max(heatmap)  # Normalize

    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)
    return superimposed_img

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
    st.write("Classifying...")
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_resized = tf.image.resize(img, (256, 256))
    img_resized = img_resized / 255.0
    img_resized = np.expand_dims(img_resized, axis=0)
    img_resized = tf.convert_to_tensor(img_resized, dtype=tf.float32)
    
    prediction = model.predict(img_resized)
    confidence = float(prediction[0][0]) * 100
    result = "Multiple Sclerosis" if prediction > 0.5 else "Healthy"
    
    # Generate Grad-CAM
    print(model.summary())
    print([layer.name for layer in model.layers if 'conv' in layer.name])

    heatmap = make_gradcam_heatmap(img_resized, model, "conv2d_5")  # Change from "conv2d_4" to "conv2d_5"
    gradcam_img = overlay_heatmap(img, heatmap)
    
    # Save Grad-CAM image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        cv2.imwrite(temp_file.name, gradcam_img)
        gradcam_path = temp_file.name
    
    col1, col2 = st.columns(2)
    col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    col2.write(f"Prediction: **{result}**\nConfidence: **{confidence:.2f}%**")
    
    st.subheader("Grad-CAM Visualization")
    st.image(gradcam_path, caption="Grad-CAM Heatmap", use_column_width=True)
