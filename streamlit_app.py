import streamlit as st
from PIL import Image


st.title('🤖 machine learning app')

st.write('facial emotion classification using tensorflow and transfer learning')
st.image('angry.jpeg', caption="an angry person")


# Title of the app
st.title("Image Upload and Display")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the image using PIL
    image = Image.open(uploaded_file)
    
    # Display the image
    st.image(image, caption='Uploaded Image')
