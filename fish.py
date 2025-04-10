import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the best model
model = tf.keras.models.load_model('C:/Users/ADMIN/Desktop/Projects/model_mobilenet.h5')  # Change to your saved model path

class_names = ['animal fish',
 'animal fish bass',
 'fish sea_food black_sea_sprat',
 'fish sea_food gilt_head_bream',
 'fish sea_food hourse_mackerel',
 'fish sea_food red_mullet',
 'fish sea_food red_sea_bream',
 'fish sea_food sea_bass',
 'fish sea_food shrimp',
 'fish sea_food striped_red_mullet',
 'fish sea_food trout'] 


# Streamlit interface
st.title("Fish Classification App")
st.write("Upload an image to classify the fish.")

# Image upload
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    img = image.load_img(uploaded_image, target_size=(256, 256))
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  

    # Predict
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    # Display prediction
    st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)
    st.write(f"Prediction: {class_names[class_idx]}")
    st.write(f"Confidence: {confidence * 100:.2f}%")











