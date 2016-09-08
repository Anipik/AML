  # -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import scipy
import numpy as np
import argparse
import os

#parser = argparse.ArgumentParser(description='Decide if an image is a picture of a bird')
#parser.add_argument('image', type=str, help='The image image file to check')
#args = parser.parse_args()


# Same network definition as before
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.load("train1_models/driver-classifier.tfl")

# Load the image file
#img = scipy.ndimage.imread(args.image, mode="RGB")

# Scale it to 32x32
#img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')

# Predict
# prediction = model.predict([img])

# # Check the result.
# print(prediction)
# is_bird = np.argmax(prediction[0]) == 1

# if is_bird:
#     print("That's a bird!")
# else:
#     print("That's not a bird!")
i=0
path = "test"
import csv

with open('sample_submission1.csv', 'w') as csvfile:
    fieldnames = ['img', 'c0' , 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    rowwriter = csv.writer(csvfile)
    for filename in os.listdir(path):
		img = scipy.ndimage.imread(os.path.join(path,filename),mode="RGB")
		img = scipy.misc.imresize(img, (32, 32), interp="bicubic").astype(np.float32, casting='unsafe')
		prediction = model.predict([img])
		mydict = {}
		mydict['img']=filename
		for k in range(0,10):
			mydict[fieldnames[k+1]]=prediction[0][k]
		#print(i)
		writer.writerow(mydict)
		# if i == 100:
		# 	break
		i=i+1
		print(i)
		#writer.writerow([filename prediction[0]])

