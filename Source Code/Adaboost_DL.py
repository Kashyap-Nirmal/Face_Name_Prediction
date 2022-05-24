#######################################################################
# This code block contains all the necessary imports.
# This code was executed on Google Colaboratory.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from numpy import asarray
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, ConfusionMatrixDisplay
from PIL import Image
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Flatten

import keras
import matplotlib.pyplot as plot
import seaborn as sns
import numpy as np
import random
import cv2
import os
import glob
import math

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

#######################################################################
'''
  The dataset directory structure should be "train" directory which contains all the training classes as subdirectories. 
  And "test" directory which contains all the testing classes as subdirectories. 
  Image resolution used is 64 * 64.

  /DATASET/train/Krishna/Krishna_1.jpg
  /DATASET/train/Rahul/Rahul_1.jpg

  /DATASET/test/Krishna/Krishna_1.jpg
  /DATASET/test/Rahul/Rahul_1.jpg

'''
training_files = []
#INSERT THE TRAINING DATASET PATH HERE.
for filename in glob.glob(r"/content/drive/MyDrive/Name_Dataset/Face_Crop - Train Test/train" + "/**/*", recursive=True): #assuming gif
    training_files.append(filename)

testing_files = []
#INSERT THE TRAINING DATASET PATH HERE.
for filename in glob.glob(r"/content/drive/MyDrive/Name_Dataset/Face_Crop - Train Test/test" + "/**/*", recursive=True): #assuming gif
    testing_files.append(filename)

#######################################################################

random.shuffle(training_files)
print(len(training_files))
print(len(testing_files))

#######################################################################

#INSERT THE CLASS LABELS HERE.
class_labels = {"Krishna":0, "Pranav":1, "Priya":2, "Rahul":3, "Sonal": 4}

#######################################################################

training_image = []
testing_image = []
training_label = []
testing_label = []

for imgX in training_files:
  try:
    image = cv2.imread(imgX)
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    training_image.append(image)
    '''
    Suppose below is the naming convention used. We want class label 'Krishna'. So we will extract last second element of the list.
      C:\Files\gender_dataset_face\Krishna\Krishna_1.jpg.
    '''
    label = imgX.split(os.path.sep)[-2]     
    l_index = class_labels[label]
        
    training_label.append([l_index]) # [[1], [0], [0], ...]

  except:
    print("a")
    
for imgX in testing_files:
  try:
    image = cv2.imread(imgX)
    image = cv2.resize(image, (64, 64))
    image = img_to_array(image)
    testing_image.append(image)
    
    label = imgX.split(os.path.sep)[-2] # C:\Files\gender_dataset_face\woman\face_1162.jpg
    l_index = class_labels[label]
        
    testing_label.append([l_index]) # [[1], [0], [0], ...]
  except:
    print("b")

#######################################################################

# load model without classifier layers
base_model = VGG16(include_top=False, weights='imagenet', input_shape=(64, 64, 3), classes=5)

flat1 = Flatten()(base_model.layers[-1].output)
class1 = Dense(4096, activation='relu')(flat1)

modelX = Model(inputs=base_model.inputs, outputs=class1)

#######################################################################

modelX.summary()

#######################################################################

training_image = np.array(training_image, dtype="float")/255.0
testing_image = np.array(testing_image, dtype="float")/255.0

#######################################################################

training_label = np.array(training_label)
testing_label = np.array(testing_label)

#######################################################################

training_image.shape

#######################################################################

trainX_deep = modelX.predict(training_image)
testX_deep = modelX.predict(testing_image)

#######################################################################

svc = SVC(probability=True, kernel="linear")
adaboost = AdaBoostClassifier(n_estimators=3, base_estimator=svc, learning_rate=0.5, algorithm='SAMME.R')

#######################################################################

clfmodel = adaboost.fit(trainX_deep, training_label)

#######################################################################

label_predicted = clfmodel.predict(testX_deep)
print("Accuracy:",metrics.accuracy_score(testing_label, label_predicted))

#######################################################################

confusionmatrix = confusion_matrix(testing_label, label_predicted, normalize = 'true')  
axX = sns.heatmap(confusionmatrix , cmap = 'Blues', annot = True , cbar = True , fmt ='.2%')
axX.figbox
# Show all ticks and label them with the respective list entries
list1 = []
for key in class_labels.keys():
  list1.append(key)
axX.xaxis.set_ticklabels(list1)
axX.yaxis.set_ticklabels(list1) 
# Rotate the tick labels and set their alignment.
plot.setp(axX.get_xticklabels() , rotation=45 , ha="right", rotation_mode="anchor")
axX.set_title('Confusion Matrix for Name Prediction\n\n');
plot.show()

#######################################################################

print(classification_report(testing_label, label_predicted))

#######################################################################