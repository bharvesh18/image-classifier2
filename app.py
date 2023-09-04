import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/image_classifier.h5')

# Define a function to preprocess input image
def preprocess_image(image):
    img = image.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.title("Image Classifier with Streamlit")

    # File upload widget
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Preprocess and predict when an image is uploaded
        img = Image.open(uploaded_image)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        predicted_label = class_names[predicted_class]

        # Display the image and prediction
        st.image(img, caption="Uploaded Image", use_column_width=True)
        st.write(f"Prediction: {predicted_label}")

if __name__ == '__main__':
    main()
