import csv
import cv2
import numpy as np

def process_image(fileline):
  filepath = './training_data/IMG/' + fileline.split("/")[-1]
  image = cv2.imread(filepath)
  return image

lines = []
with open('./training_data/driving_log.csv') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    lines.append(line)
    # print("this is line: ", line[0])


images = []
measurements = []
for line in lines:
  # print("this is filename: ", filename)
  center_image = process_image(line[0])
  left_image = process_image(line[1])
  right_image = process_image(line[2])

  center_measurement = float(line[3])
  correction = 0.2
  left_measurement = center_measurement + correction
  right_measurement = center_measurement - correction

  flipped_center_image = np.fliplr(center_image)
  flipped_left_image = np.fliplr(left_image)
  flipped_right_image = np.fliplr(right_image)

  flipped_center_measurement = -center_measurement
  flipped_left_measurement = -left_measurement
  flipped_right_measurement = -right_measurement

  images.extend([center_image, left_image, right_image, flipped_center_image, flipped_left_image, flipped_right_image])
  measurements.extend([center_measurement, left_measurement, right_measurement, flipped_center_measurement, flipped_left_measurement, flipped_right_measurement])

X_train = np.array(images)
y_train = np.array(measurements)

# print("this is X_train shape: ", X_train.shape)
# print("this is y_train shape: ", y_train.shape)

# model training
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3])

model = Sequential()

# preprocessing, normalizing (/255) and mean centering (-.5)
model.add(Cropping2D(cropping=((60,25), (0,0)), input_shape=input_shape))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

# 1st CNN layer
nb_filters = 60
filter_size = (3,3)
model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1], border_mode='valid'))
pool_size = (2,2)
model.add(MaxPooling2D(pool_size=pool_size))
# model.add(Dropout(.5))

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
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1)) #single node representing steering angle, unlike classification, which has # of final nodes equal to number of classes

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=5, validation_split=0.2, shuffle=True)

model.save('model.h5')
