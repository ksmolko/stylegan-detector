import sys

if len(sys.argv) != 2:
    print("Usage: test_script.py [test_directory]")
    sys.exit()

from keras import models
from keras.models import load_model
import numpy as np
from PIL import Image as PIL_Image
import os
import tensorflow as tf
import cv2
from skimage.feature import greycomatrix


# use getCoMat for predict input
def getCoMatrices(img):
    #rgb = img.split()
    #print(cv2.split(img))
    b,g,r = cv2.split(img)
    distance = 1 # distance offset to detect pixel pairs
    angle = 0 # angle offset to detect distance pairs (0 is to the right, np.pi/2 is up)
    rcomatrix = greycomatrix(r, [distance], [angle])
    gcomatrix = greycomatrix(g, [distance], [angle])
    bcomatrix = greycomatrix(b, [distance], [angle])
    tensor = tf.constant([rcomatrix[:,:,0,0], gcomatrix[:,:,0,0], bcomatrix[:,:,0,0]])
    tensor = tf.reshape(tensor, [256, 256, 3])
    return tensor


###### Main code ######

png_model = load_model('models/png_model/ImageDetectmodel.h5')
jpg_model = load_model('models/jpg_model/ImageDetectmodel.h5')
print("Model loaded")

TEST_PATH = sys.argv[1]
if not (TEST_PATH.endswith("/") or TEST_PATH.endswith("\\")):
    TEST_PATH = TEST_PATH + "/"
files = os.listdir(TEST_PATH)

print("Predicting...")

numReal = 0
numFake = 0

for i in files:
    # Ignore .DS_Store for those on Mac
    if i == ".DS_Store":
        continue

    img = cv2.imread(TEST_PATH + i)
    img = cv2.resize(img, (1024, 1024))
    comat = getCoMatrices(img)
    comat = tf.reshape(comat, [1, 256, 256, 3])
    
    if i.endswith(".png"):
        prediction = np.argmax(png_model.predict(comat), axis=-1)
    elif i.endswith(".jpg"):
        prediction = np.argmax(jpg_model.predict(comat), axis=-1)
    else:
        print("Error: " + i + " is neither a PNG nor JPEG")
    
    if prediction == 0:
        print(i + ": " + "Fake")
        numFake += 1
    else:
        print(i + ": " + "Real")
        numReal += 1

print("# Real: " + str(numReal))
print("# Fake: " + str(numFake))
