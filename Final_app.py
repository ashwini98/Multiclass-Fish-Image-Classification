import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# First Streamlit command
st.set_page_config(page_title="Fish Classifier")

# Load model with caching
@st.cache_resource
def load_model():
    st.write("Loading model...")  # Debugging line
    return tf.keras.models.load_model("models/best_MobileNet.keras")

model = load_model()

class_labels = [
    'animal fish',
    'animal fish bass',
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

def preprocess_image(image: Image.Image):
    """Preprocess the image for model prediction."""
    image = image.resize((224, 224))  # Resize image to match input size of MobileNet
    image = np.array(image) / 255.0   # Normalize the image
    if image.shape[-1] == 4:  # If the image has an alpha channel (RGBA), remove it
        image = image[..., :3]
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Streamlit app interface
st.title("🐟 Fish Image Classifier")
st.write("Upload an image of a fish or seafood item to classify it.")

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Image uploaded successfully!")  # Debugging line

    with st.spinner("Classifying..."):
        # Preprocess the image and make a prediction
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)[0]

        # Get the top 3 predictions
        top3 = predictions.argsort()[-3:][::-1]

        # Display the top 3 predictions
        st.subheader("Top Predictions:")
        for i in top3:
            st.write(f"**{class_labels[i]}**: {predictions[i]*100:.2f}%")

        # Display the confidence score of the top prediction
        st.subheader("Top Prediction:")
        top_label = class_labels[top3[0]]
        top_score = predictions[top3[0]] * 100
        #st.write(f"**{top_label}**: {top_score:.2f}%")
        st.write(f"**{top_label}**")

# Error handling 
else:
    st.write("Please upload a fish image to get a prediction.")
