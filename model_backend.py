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

# MS-Quiz-Feature & News-Feature

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf

# Load the trained model
model = load_model('models/imageclassifier.h5')

st.title("Multiple Sclerosis MRI Prediction")

# Tabs for MRI classification, Risk Assessment Quiz, and Educational Resources
tab1, tab2, tab3 = st.tabs(["MRI Prediction", "Symptoms & Risk Assessment Quiz", "Educational Resources"])

with tab1:
    st.header("Upload an MRI Image")
    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)
        st.write("Classifying...")

        # Preprocess the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_resized = tf.image.resize(img, (256, 256)) / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)

        # Make prediction
        prediction = model.predict(img_resized)
        result = "Multiple Sclerosis" if prediction > 0.5 else "Healthy"

        col1, col2 = st.columns(2)
        col1.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        col2.write(f"Prediction: **{result}**")

with tab2:
    st.header("Symptoms & Risk Assessment Quiz")
    st.write("Answer the following questions to assess your risk:")

    # Questions (Yes = 1, No = 0)
    q1 = st.radio("Do you often experience fatigue?", ("No", "Yes"))
    q2 = st.radio("Have you noticed muscle weakness or spasms?", ("No", "Yes"))
    q3 = st.radio("Do you have frequent vision problems (blurriness, double vision)?", ("No", "Yes"))
    q4 = st.radio("Have you experienced balance or coordination issues?", ("No", "Yes"))
    q5 = st.radio("Do you often feel numbness or tingling in your limbs?", ("No", "Yes"))
    q6 = st.radio("Do you have difficulty with memory or concentration?", ("No", "Yes"))

    if st.button("Calculate Risk Score"):
        risk_score = sum([q1 == "Yes", q2 == "Yes", q3 == "Yes", q4 == "Yes", q5 == "Yes", q6 == "Yes"]) / 6 * 100
        st.write(f"Your MS Risk Score: **{risk_score:.2f}%**")
        
        if risk_score > 50:
            st.warning("Your symptoms suggest a moderate to high risk. Consider consulting a healthcare professional.")
        else:
            st.success("Your risk level appears low. However, if you have concerns, seek medical advice.")

with tab3:
    st.header("Educational Resources")
    st.subheader("Articles About MS and Treatments")
    st.markdown("- [Mayo Clinic - Multiple Sclerosis Overview](https://www.mayoclinic.org/diseases-conditions/multiple-sclerosis/symptoms-causes/syc-20350269)")
    st.markdown("- [NIH - MS Research](https://www.ninds.nih.gov/Disorders/All-Disorders/Multiple-Sclerosis-Information-Page)")
    st.markdown("- [National MS Society - MS Information](https://www.nationalmssociety.org/What-is-MS)")

    st.subheader("Videos on Multiple Sclerosis")
    st.video("https://www.youtube.com/watch?v=ScaY3P2UOz8")  # Example YouTube video
    st.video("https://www.youtube.com/watch?v=Z1ibVlGflPs")

    st.subheader("Trusted Organizations")
    st.markdown("- [Mayo Clinic](https://www.mayoclinic.org/)")
    st.markdown("- [NIH - National Institutes of Health](https://www.nih.gov/)")
    st.markdown("- [National MS Society](https://www.nationalmssociety.org/)")


# Commands to run 
#  1111  python3 -m venv myenv
#  1112  source myenv/bin/activate
#  1113  pip install streamlit tensorflow numpy opencv-python
#  1115  streamlit run model_backend.py 
#  1116  deactivate
