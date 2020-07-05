import os
import csv


# read csv file and put into a list
data_folder = './data/'
samples = []
with open(data_folder + 'driving_log.csv') as csvfile:
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

angle_correction = 0.2

# data generator
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset : offset+batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples:
				for i in range(3): # add center, left, right images
					img_name = data_folder + 'IMG/' + batch_sample[i].split('/')[-1]
					image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
					images.append(image)
					
					center_angle = float(batch_sample[3]) # str to float
					# +/- correction for left/right camera
					if (i == 0):
						angles.append(center_angle)
					elif (i == 1):
						angles.append(center_angle + angle_correction)
					else:
						angles.append(center_angle - angle_correction)

					# data augmentation by flipping image
					images.append(np.fliplr(image))
					# +/- correction for left/right camera, then flip
					if (i == 0):
						angles.append(center_angle * -1)
					elif (i == 1):
						angles.append((center_angle + angle_correction) * -1)
					else:
						angles.append((center_angle - angle_correction) * -1)

			X_train = np.array(images)
			y_train = np.array(angles)
			yield shuffle(X_train, y_train)

# Set our batch size
batch_size = 32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)


input_shape = (160, 320, 3) # Original image format
row, col, ch = 80, 320, 3 # Trimmed image format
keep_prob = 0.5

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
model.add(Dense(100, activation='relu'))
model.add(Dropout(keep_prob)) # add Dropout layer to reduce overfitting
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

from math import ceil

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
			steps_per_epoch=ceil(len(train_samples)/batch_size), 
			validation_data=validation_generator, 
			validation_steps=ceil(len(validation_samples)/batch_size), 
			epochs=5, verbose=1)

model.save('model.h5')
print('Model Saved.')
model.summary()


import matplotlib.pyplot as plt

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('error_plot.jpg', dpi=400)