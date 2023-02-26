
# import libraries
import streamlit as st
import numpy as np
from tensorflow.image import decode_image, resize
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

labels_list = ['apple', 'avocado', 'banana', 'kiwi', 'lemon', 'orange']

# add title and instructions
st.title("Fruit classification model (Banana,Avocado,Banana,Kiwi,Lemon,Orange)")
model = load_model('models/fruits_cnn_vgg.h5')

@st.cache(allow_output_mutation=True)
def load_model(model_filename):
    # load model
    model = load_model(model_filename)
    
    return model

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        img = decode_image(
            image_data,
            channels=3,
            name=None,
            expand_animations=False
        )
        img = resize(img,[224,224])
        return img.numpy()
        
    else:
        return None

# image pre-processing function

def preprocess_image(img):
    image = np.expand_dims(img, axis = 0)
    image = preprocess_input(image)
    
    return image

# image prediction function

def make_prediction(img):
    image = preprocess_image(img)
    class_probs = model.predict(image)
    predicted_class = np.argmax(class_probs)
    predicted_label = labels_list[predicted_class]
    predicted_prob = class_probs[0][predicted_class]

    return predicted_label, predicted_prob

def main():
    
    img = load_image()
    if img is None:
        st.text("Waiting for upload...")
    else:
        predicted_label, predicted_prob = make_prediction(img)
        st.text(f"The image is classified as {predicted_label} with a probability of {predicted_prob}")

if __name__ == '__main__':
    main()
