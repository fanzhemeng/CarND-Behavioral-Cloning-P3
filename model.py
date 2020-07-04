import os
import csv


# read csv file and put into a list
samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader, None)
	for line in reader:
		samples.append(line)

# split training and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


# data generator
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		half_batch_size = int(batch_size / 2) # use half size cause we are adding flipped data into batch too
		for offset in range(0, num_samples, half_batch_size):
			batch_samples = samples[offset : offset+half_batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				name = '/opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3]) # str to float
				images.append(center_image)
				angles.append(center_angle)
				# data augmentation by flipping images and steering measurements
				images.append(np.fliplr(center_image))
				angles.append(0-center_angle)

			# trim image to only see section with road
			X_train = np.array(images)
			y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


input_shape = (160, 320, 3) # Original image format
row, col, ch = 80, 320, 3 # Trimmed image format


from keras.models import Sequential
from keras.layers import Lambda, Cropping2D, Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
# Preprocess incoming data, first crop useless parts, then centered around zero with small stddev
model.add(Cropping2D(cropping=((60,20), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
#finish defining the rest of your model architecture here
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

from math import ceil

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
			steps_per_epoch=ceil(len(train_samples)/batch_size), 
			validation_data=validation_generator, 
			validation_steps=ceil(len(validation_samples)/batch_size), 
			epochs=5, verbose=1)

model.save('model.h5')


import matplotlib.pyplot as plt

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()