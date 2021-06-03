# StyleGAN Detector

Our StyleGAN detector is a neural network and python script used to detect if an image of a human face is an image that is generated using a StyleGAN or not.

## Dependencies
These sets of scripts require the following to run:
* Python 3.8.6 (This will NOT work with Python 3.9, as Tensorflow does not support this version yet as of the time of submission)
* Tensorflow 2.3.0
* Keras 2.4.3
* OpenCV 4.1.2
* Scikit-image 0.16.2
* Numpy 1.18.5
* Matplotlib 3.3.3
* Pillow 8.0.1

We expect that these scripts should work with slightly different versions of the above packages, but no guarantees are made.

## Usage

### Training

#### Dataset

The datasets used for training were very large, and as such, have not been committed with this code. If you wish to train, you can compile your own dataset, or use the datasets that we used at the following links:

Fake Images: https://drive.google.com/drive/folders/1-5HnXJuN1ofCrCSbbVSH3NnP62BZTh4s
Real Images: https://drive.google.com/drive/folders/1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

After you do this, you must extract all the real images and place them in one folder with NO subfolders, and likewise for the fake images.

We have also supplied a `augment.py` script that will augment the images in your dataset to create more images, in case the dataset may be too small for your needs.

#### train_png_model.py

After you have the directory of real and fake images, run the training script with the following command:
```
python train_png_model.py [fake img directory] [real img directory]
```
Output may remain blank for a long time while the program loads all the co-occurrence matrices of each image into memory. If the user does not have a sufficient amount of memory to store all the images, this program will crash. After the program finishes, it will display plots of the results, and print out the final validation accuracy. It will also save the model to the same directory as the script.

#### train_jpg_model.py

To train a JPEG model, follow the same instructions as with the PNG model, but this time with JPEG images. Note that this training script is tuned for JPEG images with a quality factor of 75. We have supplied a helper script that we wrote called `PNG_to_JPG.py` that will convert all the PNG images in one folder to JPEG, and then place them in another folder. To run this script, execute the following command:
```
python PNG_to_JPG.py [source dir] [destination dir]
```
The output of this script will be the same as with the PNG training script.

### Testing

#### test_script.py

Create a directory where you will host all the images you want to test, and then run the following:
```
python test_script.py [test_directory]
```
The program will output the prediction for each image, and then display the number of Real and Fake images detected in the directory. Please note that we have only trained models for PNG images and JPEG images with a quality factor of 75. This script will not work with images that are neither PNG nor JPEG.
