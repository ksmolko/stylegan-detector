import glob
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from matplotlib import pyplot as plt# to show images
import cv2

def getPath(pathName):
  PathList = glob.glob(pathName + '/*.png') + glob.glob(pathName + '/*.jpg')
  return PathList


def main():
  dirList = getPath('realImageLarge')
  path = 'realAugmented/'

  i = 0

  for im in dirList:
    img = cv2.imread(im)

    # convert to gray scale
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    RGB_grayImage = cv2.cvtColor(grayImage, cv2.COLOR_BGR2RGB)
    grayPath = path + str(i) + 'grayImage.jpg'
    cv2.imwrite(grayPath, grayImage)

    # smoothing images with a gaussian filter
    # smoothedImage = cv2.GaussianBlur(img,(11,11),5)
    # RGB_smoothedImage = cv2.cvtColor(smoothedImage, cv2.COLOR_BGR2RGB);

    # brighten image
    datagen1 = ImageDataGenerator(brightness_range=[1.0,2])
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    it = datagen1.flow(np.expand_dims(RGB_img, 0), batch_size=1)
    batch = it.next()
    RGB_brightImage = batch[0].astype('uint8')
    RGB_brightImagePath = path + str(i) + 'RGB_brightImage.jpg'
    cv2.imwrite(RGB_brightImagePath, RGB_brightImage)

    # darken an Image
    datagen2 = ImageDataGenerator(brightness_range=[0.25,1.0])
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    it = datagen2.flow(np.expand_dims(RGB_img, 0), batch_size=1)
    batch = it.next()
    RGB_darkImage = batch[0].astype('uint8')  
    RGB_darkImagePath = path + str(i) + 'RGB_darkImage.jpg'
    cv2.imwrite(RGB_darkImagePath, RGB_darkImage)
    
    print(i)
    print(grayPath)
    print(RGB_brightImagePath)
    print(RGB_darkImagePath)
    i+=1

  print('Done running')

main()