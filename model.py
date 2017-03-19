import csv
import cv2
import numpy as np

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
  # print("this is image: ", image.shape)
  images.append(image)

  measurement = float(line[3])
  # print("this is measurement: ", measurement)
  measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

print("this is X_train shape: ", X_train.shape)
print("this is y_train shape: ", y_train.shape)

# model training
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers.convolutional import Convolution2D

input_shape = (160,320,3)
model = Sequential()

nb_filters = 60
filter_size = (3,3)

model.add(Convolution2D(nb_filters, filter_size[0], filter_size[1], border_mode='valid', input_shape=input_shape))
model.add(Flatten())
model.add(Dense(1)) #single node representing steering angle

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, nb_epoch=20, validation_split=0.2, shuffle=True)

model.save('model.h5')
