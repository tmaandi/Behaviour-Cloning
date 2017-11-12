import os
import numpy as np
import csv
from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
import cv2
from generator import generator
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


# Loading and initializing Data
img_path = "./example_data/IMG/"
csv_path = "./example_data/"
samples = []

with open(csv_path + "driving_log.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for line in reader:
    	samples.append(line)

# Splitting the data into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Using generator to process data into batches without running into 
# memory shortage error
train_generator = generator(train_samples, batch_size = 32, path = img_path)
validation_generator = generator(validation_samples, batch_size =32, path = img_path)

# Keras NN Model
# Number of epochs
epochs = 4


model = Sequential()
# Cropping Layer
# Cropping top 50 rows and bottom 20 rows
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))

# Normalization layer
model.add(Lambda(lambda x: (x/255.0) - 0.5))

# NVIDIA-like Architecture
model.add(Convolution2D(24,5,2,activation='relu', subsample=(2,2)))
model.add(Convolution2D(36,5,2,activation='relu', subsample=(2,2)))
model.add(Convolution2D(48,5,2,activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,1,activation='relu', subsample=(2,2)))
model.add(Convolution2D(64,3,1,activation='relu', subsample=(2,2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.40))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compiling the CNN Model
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= \
	4*len(train_samples), validation_data=validation_generator, \
	nb_val_samples=len(validation_samples), nb_epoch= epochs, verbose=1)

# Visualizing loss

# Print keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch

# Disable plotting error on AWS Instance
plt.switch_backend('agg')

fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc = 'upper right')
plt.show()

print("Saving figure")
fig.savefig('Loss_and_Validation.png')
print('Saving Model.h5\n')
model.save('model.h5')
print('Model.h5 saved\n')