#  importing libraries
import os
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import pickle

#  configuring device
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print('Running on the GPU')
else:
    device = torch.device('cpu')
    print('Running on the CPU')


#  building neural network (100px with batchnorm)
class CarRecognition(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.conv7 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(8192, 514)
        self.fc2 = nn.Linear(514, 128)
        self.fc3 = nn.Linear(128, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.pool7 = nn.MaxPool2d(2, 2)
        self.batchnorm_conv1 = nn.BatchNorm2d(32)
        self.batchnorm_conv2 = nn.BatchNorm2d(32)
        self.batchnorm_conv3 = nn.BatchNorm2d(64)
        self.batchnorm_conv4 = nn.BatchNorm2d(64)
        self.batchnorm_conv5 = nn.BatchNorm2d(128)
        self.batchnorm_conv6 = nn.BatchNorm2d(128)
        self.batchnorm_conv7 = nn.BatchNorm2d(128)
        self.batchnorm_fc1 = nn.BatchNorm1d(514)
        self.batchnorm_fc2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.view(-1, 3, 100, 100).float()
        x = F.relu(self.batchnorm_conv1(self.conv1(x)))
        x = self.pool2(F.relu(self.batchnorm_conv2(self.conv2(x))))
        x = F.relu(self.batchnorm_conv3(self.conv3(x)))
        x = self.pool4(F.relu(self.batchnorm_conv4(self.conv4(x))))
        x = F.relu(self.batchnorm_conv5(self.conv5(x)))
        x = F.relu(self.batchnorm_conv6(self.conv6(x)))
        x = self.pool7(F.relu(self.batchnorm_conv7(self.conv7(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.batchnorm_fc1(self.fc1(x)))
        return x


def save_image(img):
    #  saving uploaded image
    with open('image.jpg', 'wb') as f:
        f.write(img.getbuffer())
    pass


def compute_similarities(image):
    #  loading model state
    model = CarRecognition()
    model.load_state_dict(torch.load('app_files/model_state100.pt', map_location=device))

    #  loading image features
    with open('app_files/similarity_features.pkl', 'rb') as f:
        extracted_features = pickle.load(f)

    #  processing image
    img = cv2.imread(image)
    img = cv2.resize(img, (100, 100))
    img = img/255
    img = transforms.ToTensor()(img)

    #  extracting image features
    model.eval()
    with torch.no_grad():
        img = model(img)

    #  computing similarities
    similarity_scores = [[F.cosine_similarity(img, im).item(), f] for im, f in extracted_features]

    #  extracting scores and filenames
    scores = [x[0] for x in similarity_scores]
    filenames = [x[1] for x in similarity_scores]

    #  creating series of scores
    score_series = pd.Series(scores, index=filenames)
    score_series = score_series.sort_values(ascending=False)

    #  extracting recommendations
    recommendations = score_series.index[:4]
    recommendations = [os.path.join('images/similarity_images', f) for f in recommendations]

    #  creating captions
    labels = score_series.values[:4]
    labels = [round(x*100) for x in labels]
    labels = [f'similarity: {x}%' for x in labels]

    #  extracting  highest recommendation scores
    recommendation_check = score_series.values[0]

    #  deleting image
    os.remove('image.jpg')

    print(score_series.head(4))

    return recommendations, labels, recommendation_check


def output(uploaded):
    try:
        #  saving image
        save_image(img=uploaded)

        #  provide feedback
        st.write('##### Image uploaded!')

        #  display image
        st.image('image.jpg', width=365)

        #  derive recommendation
        recommended, captions, check = compute_similarities(image='image.jpg')

        #  feedback
        st.write('#### Output:')

        #  check recommendations
        if check < 0.50:
            #  feedback
            st.error(
                """
                Hmmm there appears to be no similar images in storage.
                Either that or this might be a unique looking car or not a car at all.
                Would you still be interested in seeing what the model thinks about this image regardless? 
                Note however that the recommendations will be very dissimilar.
                """
            )

            #  receive user option
            option = st.checkbox('show me what the model thinks')

            if option:
                #  feedback
                st.success('Here are the most likely images...')
                #  displaying likely images
                st.image(recommended, caption=captions, width=300)

        else:
            #  feedback
            st.success('Here are some similar images...')
            #  displaying similar images
            st.image(recommended, caption=captions, width=300)

    except AttributeError:
        #  handle exception
        st.write('Please upload image')
    pass
