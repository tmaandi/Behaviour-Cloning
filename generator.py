import sklearn
import cv2
import numpy as np

def generator(samples, batch_size=32, path = './example_data/IMG/'):
	num_samples = len(samples)
	while 1:
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size] 

			images = []
			angles = []
			for batch_sample in batch_samples:
				name = path + batch_sample[0].split('/')[-1]
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)

			# trim image to only see section with road
			X_train = np.array(images)
			Y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, Y_train)