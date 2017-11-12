import matplotlib.image as mpimg
import numpy as np 
import csv
import sklearn
import matplotlib.pyplot as plt

img_path = "./example_data/IMG/"
csv_path = "./example_data/"
samples = []

with open(csv_path + "driving_log.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader, None)  # skip the headers
    for line in reader:
    	samples.append(line)

sklearn.utils.shuffle(samples)

image_center_name = img_path + samples[0][0].split('/')[-1]
image_center = mpimg.imread(image_center_name)

image_left_name = img_path + samples[0][1].split('/')[-1]
image_left = mpimg.imread(image_left_name)

image_right_name = img_path + samples[0][2].split('/')[-1]
image_right = mpimg.imread(image_right_name)

image_center_flipped = np.fliplr(image_center)

fig1 = plt.figure()

plt.subplot(3,1,1)
plt.imshow(image_left)
plt.title('Left Camera')

plt.subplot(3,1,2)
plt.imshow(image_center)
plt.title('Center Camera')

plt.subplot(3,1,3)
plt.imshow(image_right)
plt.title('Right Camera')
plt.tight_layout()

fig1.savefig('Left_Right_and_Center')
plt.show()

fig2 = plt.figure()
plt.subplot(2,1,1)
plt.imshow(image_center)
plt.title('Center Image')
plt.subplot(2,1,2)
plt.imshow(image_center_flipped)
plt.title('Flipped Center Image')
plt.tight_layout()
fig2.savefig('Flipped')
plt.show()
