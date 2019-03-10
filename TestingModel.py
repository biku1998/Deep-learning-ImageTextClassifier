import os
import sys
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss, confusion_matrix
import matplotlib.pyplot as plt
from Models import SigmoidNeuron

LEVEL = 'level_1'

def read_all(folder_path, key_prefix=""):
    '''
    It returns a dictionary with 'file names' as keys and 'flattened image arrays' as values.
    '''
    print("Reading:")
    images = {}
    files = os.listdir(folder_path)
    for i, file_name in tqdm(enumerate(files), total=len(files)):
        file_path = os.path.join(folder_path, file_name)
        image_index = key_prefix + file_name[:-4]
        image = Image.open(file_path)
        image = image.convert("L")
        images[image_index] = np.array(image.copy()).flatten()
        image.close()
    return images

def print_accuracy(sn):
    Y_pred_train = sn.predict(X_scaled_train)
    Y_pred_binarised_train = (Y_pred_train >= 0.5).astype("int").ravel()
    accuracy_train = accuracy_score(Y_pred_binarised_train, Y_train)
    print("Train Accuracy : ", accuracy_train)
    print("-"*50)


languages = ['ta', 'hi', 'en']

images_train = read_all("dataset_train/"+LEVEL+"/"+"background", key_prefix='bgr_') # change the path
for language in languages:
  images_train.update(read_all("dataset_train/"+LEVEL+"/"+language, key_prefix=language+"_" ))
# print(len(images_train))

images_test = read_all("dataset_test/kaggle_"+LEVEL, key_prefix='') # change the path
# print(len(images_test))

X_train = []
Y_train = []
for key, value in images_train.items():
    X_train.append(value)
    if key[:4] == "bgr_":
        Y_train.append(0) # appending 0 as the image has no text.
    else:
        Y_train.append(1) # appending 1 as the image has text.

ID_test = []
X_test = []
for key, value in images_test.items():
  ID_test.append(int(key))
  X_test.append(value)
  
        
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)

# print(X_train.shape, Y_train.shape)
# print(X_test.shape)


scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)


# using cross entropy loss function.
sn_ce = SigmoidNeuron()
sn_ce.fit(X_scaled_train, Y_train, epochs=300, learning_rate=0.015,display_loss=True,loss_func = "ce")

# using cross mean-squared-error loss function.
sn_mse = SigmoidNeuron()
sn_mse.fit(X_scaled_train,Y_train,epochs = 300 , learning_rate = 0.015,display_loss = True,loss_func = "mse")

# printing the accuracy 

print_accuracy(sn_ce)

print_accuracy(sn_mse)


