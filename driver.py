# Load path/class_id image file:
from __future__ import division, print_function, absolute_import

directory = '/home/sun/Machine Learning/DrousyDriver/train'

from tflearn.data_utils import build_image_dataset_from_dir
import pickle

# Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle


##build dataset and dump
X,Y=build_image_dataset_from_dir(directory,dataset_file="my_tflearn_high_dataset.pkl",shuffle_data=True,categorical_Y=True)

# X, Y = pickle.load(open("../DrousyDriver/my_tflearn_dataset.pkl", "rb"))
# #print(len(X[0][0]))
# #print(len(Y))

# #X, Y = shuffle(X, Y)

# # Make sure the data is normalized
# img_prep = ImagePreprocessing()
# img_prep.add_featurewise_zero_center()
# img_prep.add_featurewise_stdnorm()

# # Create extra synthetic training data by flipping, rotating and blurring the
# # images on our data set.
# img_aug = ImageAugmentation()
# img_aug.add_random_flip_leftright()
# img_aug.add_random_rotation(max_angle=25.)
# img_aug.add_random_blur(sigma_max=3.)

# # Define our network architecture:

# # Input is a 32x32 image with 3 color channels (red, green and blue)
# network = input_data(shape=[None, 32, 32, 3],
#                      data_preprocessing=img_prep,
#                      data_augmentation=img_aug)


# network = conv_2d(network, 32, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = conv_2d(network, 64, 3, activation='relu')
# network = conv_2d(network, 64, 3, activation='relu')
# network = max_pool_2d(network, 2)
# network = fully_connected(network, 512, activation='relu')
# network = dropout(network, 0.5)
# network = fully_connected(network, 10, activation='softmax')


# network = regression(network, optimizer='adam',
#                      loss='categorical_crossentropy',
#                      learning_rate=0.001)

# # Wrap the network in a model object
# model = tflearn.DNN(network, tensorboard_verbose=0, checkpoint_path='train1_models1/driver-classifier1.tfl.ckpt')

# # Train it! We'll do 100 training passes and monitor it as it goes.
# model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=0.1,
#           show_metric=True, batch_size=96,
#           snapshot_epoch=True,
#           run_id='driver-classifier1')

# # Save model when training is complete to a file
# model.save("train1_models1/driver-classifier1.tfl")
# print("Network trained and saved as driver-classifier1.tfl!")