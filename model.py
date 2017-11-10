import os
import numpy as np
import csv
from keras.models import Sequential, Model 
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
import cv2
from generator import generator
# from generator_si import generator
from sklearn.model_selection import train_test_split
import matplotlib
from matplotlib import pyplot as plt
plt.switch_backend('agg')
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

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#print(train_samples[0],validation_samples[0])

train_generator = generator(train_samples, batch_size = 32, path = img_path)
validation_generator = generator(validation_samples, batch_size =32, path = img_path)

# train_generator = generator(train_samples, batch_size = 32)
# validation_generator = generator(validation_samples, batch_size =32)


# Using generator instead of the method below due to memory shortage
##########################################################
 #    next(reader, None)  # skip the headers
 #    n_samples = len(list(reader))
 #    # print(n_samples)
 #    f.seek(0) # moving pointer back to the beginning of the file
 #    next(reader, None) # skip the headers
 #    row1 = next(reader)
 #    # print(row1)

	# #reading one image 
 #    sample_img = cv2.imread(img_path + row1[0])

 #    #X_train = np.empty((n_samples, sample_img.shape[0],sample_img.shape[1],sample_img.shape[2]))
 #    Y_train = np.empty(n_samples)

 #    #print(X_train[0])
 #    print(len(Y_train))

  #   for row in reader:
  #       steering_center = float(row[3])
  #       #print(steering_center)
		# # read in images from center camera
  #       img_center = cv2.imread(img_path + row[0])
		# #print(img_center)

  #       # append to training data
  #       np.append(X_train,img_center)
  #       np.append(Y_train,steering_center)

#print(X_train, Y_train)
#########################################################

# Keras NN Model
# Number of epochs
epochs = 5

model = Sequential()

# Cropping Layer
# Cropping top 50 rows and bottom 20 rows 3,160,320
model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=(160,320,3)))

# new image shape = (3,110,320)
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
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# model.add(Convolution2D(32,3,3))
# model.add(MaxPooling2D((2,2)))
# model.add(Activation('relu'))
# model.add(Flatten())
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= \
	2*len(train_samples), validation_data=validation_generator, \
	nb_val_samples=len(validation_samples), nb_epoch= epochs, verbose=1)

# Visualizing loss

# Print keys contained in the history object
print(history_object.history.keys())

# plot the training and validation loss for each epoch
fig = plt.figure()
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set','validation set'],loc = 'upper right')
#plt.show()

print("Saving figure")
fig.savefig('Loss_and_Validation.png')
print('Saving Model.h5\n')
model.save('model.h5')
print('Model.h5 saved\n')
# Data Augumentation

# Using multiple Cameras
# with open(csv_file, 'r') as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		steering_center = float(row[3])

# 		# create adjusted steering measurements for the side camera images
# 		correction = 0.2
# 		steering_left = steering_center + correction
# 		steering_right = steering_center - correction

# 		# read in images from center, left and right cameras
# 		path = "./example_data/"
# 		img_center = process_image(np.asarray(Image.open(path + row[0])))
# 		img_left = process_image(np.asarray(Image.open(path + row[1])))
# 		img_right = process_image(np.asarray(Image.open(path + row[2])))

# 		# add images and angles to data set
# 		car_images.extend(img_center, img_left, img_right)
# 		steering_angles.extend(steering_center, steering_left,steering_right)

# Flipping images and taking opposite sign of steering measurement
#image_flipped = np.fliplr(image)
#measurement_flipped = -measurement

