import csv
import cv2
import numpy as np

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

  images.append(image)

  measurement = float(line[3])
  measurements.append(measurement)

  flipped_image = np.fliplr(image)
  images.append(flipped_image)
  flipped_measurement = -measurement
  measurements.append(flipped_measurement)

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

# 4th CNN layer
nb_filters_4 = 200
filter_size_4 = (3,3)
pool_size_4 = (2,2)
model.add(MaxPooling2D(pool_size=pool_size))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1)) #single node representing steering angle, unlike classification, which has # of final nodes equal to number of classes

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=3, validation_split=0.2, shuffle=True)

model.save('model.h5')
