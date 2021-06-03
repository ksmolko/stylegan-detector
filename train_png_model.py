import sys

if len(sys.argv) != 3:
    print("Usage: train_png_model.py [fake img directory] [real img directory]")
    sys.exit()

from matplotlib import pyplot as plt
from PIL import Image
from skimage.feature import greycomatrix 
from tensorflow.keras import datasets, layers, models
from keras.optimizers import Adam
import numpy as np
import os
import cv2
import tensorflow as tf


###### Helper functions ######

# Gets Co-occurrence matrices of image in the RGB colour channels, outputs tensor of size 3*256*256
def getCoMatrices(img):
  b,g,r = cv2.split(img)
  distance = 1 # distance offset to detect pixel pairs
  angle = 0 # angle offset to detect distance pairs (0 is to the right, np.pi/2 is up)
  rcomatrix = greycomatrix(r, [distance], [angle])
  gcomatrix = greycomatrix(g, [distance], [angle])
  bcomatrix = greycomatrix(b, [distance], [angle])
  tensor = tf.constant([rcomatrix[:,:,0,0], gcomatrix[:,:,0,0], bcomatrix[:,:,0,0]])
  tensor = tf.reshape(tensor, [256, 256, 3])
  return tensor


# Generates the dataset
def genDs(augmentedFakeFolder, augmentedRealFolder):
  trainingDs = []
  trainingLabels = []
  valDs = []
  valLabels = []
  split = 0.8 # used to split data in terms of testing and validation
  count = 0
  fakeLabel = int(0)
  realLabel = int(1)
  dirsFake = os.listdir(augmentedFakeFolder)
  dirsReal = os.listdir(augmentedRealFolder)
  imageToProcess = len(dirsFake) # Number of images in real and fake folders must be EQUAL
  maxFake = int(imageToProcess*split)
  maxReal = int(imageToProcess*split)

  # takes fake image in folder and puts into training_dataset
  # Once specified max limit hit for training_dataset puts the rest in the validation_dataset 
  for image in dirsFake:
    img = cv2.imread(os.path.join(augmentedFakeFolder, image), cv2.IMREAD_COLOR)
    if count > imageToProcess - 1:
      break
    if count > maxFake - 1:
      valDs.append(getCoMatrices(img))
      innerList = []
      innerList.append(fakeLabel)
      valLabels.append(innerList)
      count+=1
      continue
    trainingDs.append(getCoMatrices(img))
    otherInnerList = []
    otherInnerList.append(fakeLabel)
    trainingLabels.append(otherInnerList)
    count += 1 

  count = 0

  # takes real image in folder and puts into training_dataset
  # Once specified max limit hit for training_dataset puts the rest in the validation_dataset 
  for image in dirsReal:
    img = cv2.imread(os.path.join(augmentedRealFolder, image), cv2.IMREAD_COLOR)
    if count > imageToProcess - 1:
      break
    if count > maxReal - 1:
      valDs.append(getCoMatrices(img))
      innerList = []
      innerList.append(realLabel)
      valLabels.append(innerList)
      count += 1
      continue
    trainingDs.append(getCoMatrices(img))
    otherInnerList = []
    otherInnerList.append(realLabel)
    trainingLabels.append(otherInnerList)
    count += 1

  trainingLabels = np.asarray(trainingLabels)
  valLabels = np.asarray(valLabels)
  trainingDs = tf.stack(trainingDs)
  valDs = tf.stack(valDs)

  return trainingDs, trainingLabels, valDs, valLabels


# Creates the layers for the model to train
def trainModel():
  model = models.Sequential() 
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
  model.add(layers.Conv2D(32, (5, 5)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.1))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Conv2D(64, (5, 5)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.2))
  model.add(layers.Conv2D(128, (3, 3), activation='relu'))
  model.add(layers.Conv2D(128, (5, 5)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Dropout(0.25))
  model.add(layers.Flatten())
  model.add(layers.Dense(256))
  model.add(layers.Dropout(0.5))
  model.add(layers.Dense(256))
  
  model.compile(optimizer=Adam(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
  return model


# Trains the model and plots its accuracy
def plotAccuracy(model, train_matrices, train_labels, test_matrices, test_labels):
  data = model.fit(train_matrices, train_labels, epochs=50, 
                    validation_data=(test_matrices, test_labels))
  
  plt.plot(data.history['loss'])
  plt.plot(data.history['val_loss'])
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.ylim([0, 1])
  plt.xlabel('Epoch')
  plt.legend(['train loss', 'val loss'], loc='upper left')
  plt.show()


  plt.plot(data.history['accuracy'], label='accuracy')
  plt.plot(data.history['val_accuracy'], label = 'val_accuracy')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch') # Epoch is the iteration in the Neural Network
  plt.ylabel('Accuracy')
  plt.ylim([0.5, 1])
  plt.legend(loc='lower right')
  test_loss, test_acc = model.evaluate(test_matrices,  test_labels, verbose=2);
  plt.show()
  print('Accuracy is ' + str(test_acc * 100) + '%')



###### Main Script ######

# Generate dataset from images
tds, tlbl, vds, vlbl = genDs(sys.argv[1], sys.argv[2])

# Train the model
modelCNN = trainModel()
modelCNN.summary()
plotAccuracy(modelCNN, tds, tlbl, vds, vlbl)

# Save model to file
modelCNN.save_weights('ImageDetectmodel_weights.h5')
modelCNN.save('ImageDetectmodel.h5')
