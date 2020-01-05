
# import the necessary packages
import argparse
import imutils
import dlib
import cv2
import pandas as pd
import numpy as np

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True,
	help="path to input dataset")
args = vars(ap.parse_args())

# load the input image, resize it, and denoise it
print("[INFO] preprocessing faces...")
df = pd.read_csv("train.csv")
imagePaths = []
for i in df['image']:
	imagePaths.append(args["dataset"]+i)

kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1, 9,-1],
                              [-1,-1,-1]])

for (i, imagePath) in enumerate(imagePaths):
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	image = cv2.imread(imagePath)
	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	blur = cv2.GaussianBlur(image, (5, 5), 0)
	sharpened = cv2.filter2D(image, -1, kernel_sharpening)
	cv2.imwrite("preprocess/%s.png" %i, sharpened)