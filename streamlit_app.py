import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
#git clone https://github.com/parth1620/Facial-Expression-Dataset.git
#!pip install -U git+https://github.com/albumentations-team/albumentations
#pip install timm
#pip install --upgrade opencv-contrib-python

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


TRAIN_IMG_FOLDER_PATH='train/'
VAL_IMG_FOLDER_PATH='train/'

LR=0.001
BATCH_SIZE=32
EPOCHS=5

DEVICE='cuda'
MODEL_NAME='mobileNet'



from math import degrees
train_args=T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20,20)),
    T.ToTensor()
])
valid_args=T.Compose([
    T.ToTensor()
])



trainset=ImageFolder(TRAIN_IMG_FOLDER_PATH,transform=train_args)
validset=ImageFolder(VAL_IMG_FOLDER_PATH,transform=valid_args)


st.info(f"Total no. of examples in trainset : {len(trainset)}")
st.info(f"Total no. of examples in validset : {len(validset)}")

st.write(trainset.class_to_idx)

st.info('a random image')
image,label=trainset[100]
plt.imshow(image.permute(1,2,0))
plt.title(f"Label : {label}")
plt.show()
