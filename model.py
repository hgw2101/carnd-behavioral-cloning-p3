import csv
import cv2
import numpy as np

# No need to use this, this will apply a mask to the original image, but not changing
# the dimension of the image, I need to change the dimension here
def region_of_interest(img, vertices):
  mask = np.zeros_like(img)
  if len(img.shape) > 2:
    channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (255,) * channel_count
  else:
    ignore_mask_color = 255
  cv2.fillPoly(mask, vertices, ignore_mask_color)
  masked_image = cv2.bitwise_and(img, mask)
  return masked_image

lines = []
with open('./training_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    # print("this is line: ", line[0])

images = []
measurements = []
for line in lines:
  filename = line[0].split("/")[-1]
  # print("this is filename: ", filename)
  current_path = './training_data/IMG/' + filename
  image = cv2.imread(current_path)

  # cropped_image = image[60: 135, 0: 320]

  vertices = np.array([[(0,60),(0,135),(320,135),(320,60)]], dtype=np.int32)
  cropped_image = region_of_interest(image, vertices)

  images.append(cropped_image)
  measurement = float(line[3])
  measurements.append(measurement)

# cv2.imwrite('cropped_image.png', images[-1])

X_train = np.array(images)
y_train = np.array(measurements)

# print("this is X_train shape: ", X_train.shape)
# print("this is y_train shape: ", y_train.shape)

# model training
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D

input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])

model = Sequential()

# 1st CNN layer
nb_filters = 60
filter_size = (3,3)
model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1], border_mode='valid', input_shape=input_shape))
pool_size = (2,2)
model.add(MaxPooling2D(pool_size=pool_size))

# 2nd CNN layer
nb_filters_2 = 90
filter_size_2 = (3,3)
pool_size_2 = (2,2)
model.add(MaxPooling2D(pool_size=pool_size))

# 3rd CNN layer
nb_filters_3 = 120
filter_size_3 = (3,3)
pool_size_3 = (2,2)
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(1)) #single node representing steering angle, unlike classification, which has # of final nodes equal to number of classes

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True)

model.save('model.h5')
