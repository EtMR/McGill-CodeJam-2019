# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 18:21:16 2019
@author: Etermeteor
<Future Work>
# 1. Data Argumentation
# 2. Early stopping based on dev loss
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

# ----- Parameters for this script -----
base_model = 'VGG_cross_validated_retrain.h5'
save_model = 'retrained_20200506.h5'
train_path = 'NewTraining20191216'

max_epoches = 20

# ----- Load the training file into train -----
train = []

file_names1 = glob.glob('./{}/fist_*.jpg'.format(train_path))
file_names2 = glob.glob("./{}/L_*.jpg".format(train_path))
file_names3 = glob.glob("./{}/okay_*.jpg".format(train_path))
file_names4 = glob.glob("./{}/palm_*.jpg".format(train_path))
file_names5 = glob.glob("./{}/peace_*.jpg".format(train_path))
file_names = file_names1 + file_names2 + file_names3 + file_names4 + file_names5

for file in file_names:
    train.append(mpimg.imread(file))

train = np.asarray(train) # Make train into numpy array

# ----- Check the image whether it is loaded correctly ----- 
idx = random.randint(0, train.shape[0])
img = train[idx, :, :]
print("Check image number: ", idx)
plt.imshow(img) # Right now I have 100 photos * 288, 384 image

# ----- Define class index & reshape the image -----
gesture_names = {0: 'fist', 1: 'L', 2: 'okay', 3: 'palm', 4: 'peace'}
train3 = np.zeros((100, 224, 224, 3))

for i, img in enumerate(train):
    target = np.stack((img,) * 3, axis=-1)
    target = cv2.resize(target, (224, 224))
    target = target.reshape(1, 224, 224, 3).astype('float32')/255
    train3[i] = target

# ----- Check image after reshaping -----
img = train3[idx, :, :, :]
print("Image {} after resize.".format(idx))
plt.imshow(img) # Right now I have 100 photos * 224, 224 image

# ----- Generate the dataset -----
X = train3
Y = np.zeros((100, 5))
for i in list(range(100)):
    idx = i//20
    Y[i, idx] = 1

# ----- Train+Dev. / Test Separation -----
train_data, test_data, train_label, test_label = train_test_split(X, Y, test_size=0.2)
print(f'The Train Data shape : {train_data.shape}')

# ----- Load pretrained model -----
model = load_model('models/{}'.format(base_model))
model.summary() # Only the dense part is trainable, 3211392 + 16512 + 16512 + 8256 + 325 = 3252997
print('')

# ----- Compile and retrain the model -----
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
history = model.fit(train_data, train_label, 
                    validation_split=0.2, 
                    batch_size = 16, epochs = max_epoches)
loss, acc = model.evaluate(test_data, test_label)
print(f'The Accuracy on Test Data = {acc}')

# ----- Save model -----
model.save('models/{}'.format(save_model))
# Test acc = 85%