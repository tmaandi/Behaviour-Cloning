import sklearn
import matplotlib.image as mpimg
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
				center_image = mpimg.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)
				
				# appending images from left and right cameras
				correction = 0.15
				left_angle = center_angle + correction
				right_angle = center_angle - correction

				name = path + batch_sample[1].split('/')[-1]
				left_image = mpimg.imread(name)
				images.append(left_image)
				angles.append(left_angle)

				name = path + batch_sample[2].split('/')[-1]
				right_image = mpimg.imread(name)
				images.append(right_image)
				angles.append(right_angle)

				# flipping and appending center images
				center_image_flipped = np.fliplr(center_image)
				center_angle_flipped = -center_angle
				images.append(center_image_flipped)
				angles.append(center_angle_flipped)

			# trim image to only see section with road
			X_train = np.array(images)
			Y_train = np.array(angles)
			yield sklearn.utils.shuffle(X_train, Y_train)
