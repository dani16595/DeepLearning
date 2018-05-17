import matplotlib
matplotlib.use("Agg")
import constructor_modelo
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from constructor_modelo import LeNet
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os

height=80
width=80


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 450
INIT_LR = 1e-3
BS = 32
 
# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []
 
# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images("./Train")))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	if(imagePath[-4:]==".jpg" or imagePath[-4:]==".png" or imagePath[-4:]==".JPG"):
		image = cv2.imread(imagePath)
		image = cv2.resize(image, (height, width))
		image = img_to_array(image)
		data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
		label = imagePath.split(os.path.sep)[-2]
		label = 1 if label == "Pistol" else 0
		labels.append(label)


data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.25, random_state=42)

trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

print("[INFO] compiling model...")
model = LeNet.build(width=width, height=height, depth=3, classes=2)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size = BS),
	validation_data = (testX, testY), steps_per_epoch = len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

print("[INFO] serializing network...")
model.save("modelo")