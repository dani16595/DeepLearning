
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import os
from imutils import paths
import cv2
import csv, operator

height=80
width=80

imagePaths = sorted(list(paths.list_images("./Test")))

imagePaths.sort(key=lambda image: int(image.split("img")[1].split(".")[0]))

datos=[]
print("[INFO] loading network...")
model = load_model("modelo")
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	if(imagePath[-4:]==".jpg" or imagePath[-4:]==".png" or imagePath[-4:]==".JPG"):
		
		image = cv2.imread(imagePath)
		orig = image.copy()


		# pre-process the image for classification
		image = cv2.resize(image, (height, width))
		image = image.astype("float") / 255.0
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)

		# load the trained convolutional neural network

		# classify the input image
		(movil, pistola) = model.predict(image)[0]

		# build the label
		label = "Pistola" if pistola > movil else "Movil"
		if label=="Pistola":
			datos.append((imagePath[7:],0))
		else:
			datos.append((imagePath[7:],1))
csvsalida = open('salidat.csv', 'w', newline='')
salida = csv.writer(csvsalida)
salida.writerow(['ID', 'Ground_Truth'])
salida.writerows(datos)
del salida
csvsalida.close()



