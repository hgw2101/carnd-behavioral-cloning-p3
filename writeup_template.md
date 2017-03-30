#**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/model_architecture.png?token=75d872ac48854fcd74829ba5e6a1569eae90f4b2 "Model Visualization"
[image2]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/center_lane_driving.jpg?token=c2d5a731a9d9aa6b4ac34cb18a3a0a8879ba697f "Center Lane Driving"
[image3]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/recovery_1.jpg?token=1cad0d9fec8159a62fb93efa3791e432ad2e8a63 "Recovery Image"
[image4]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/recovery_2.jpg?token=8431bd3d010e612d311ee21d3f7d6adecdb7971f "Recovery Image"
[image5]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/recovery_3.jpg?token=fc005fccbddb9641b9f745c8b517a397f39a1052 "Recovery Image"
[image6]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/original.jpg?token=9e5a77d6f38f0f3973a94df504ef3776df48db7b "Normal Image"
[image7]: https://bytebucket.org/hgw2101/carnd-behavioral-cloning-p3/raw/28fd3b477c60c697f324ab74d9b64232532ca6f3/write_up_images/flipped.jpg?token=abb72b6460b9fcd95d450869688eeb3df0b127ff "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 77-102) 

The model includes RELU layers to introduce nonlinearity (code lines 80, 86, 92, 97 and 102), and the data is normalized in the model using a Keras lambda layer (code line 72). 

####2. Attempts to reduce overfitting in the model

I initially experimented with dropout by applying a Keras Dropout layer. It made the model's performance a lot worse, so I decided to try other methods to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. I initially had one lap of data, then I added another lap of data, then flipped images. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (2 laps as noted above), 1 lap of recovering from the left and right sides of the road, and 1 lap of reverse driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a multi-layered convolutional network with good preprocessing and data augmentation metrics.

However, to test that my setup works, my first step was to use a single fully-connected layer with a single output node. This, as pointed out in the project instructions, was to make sure that everything is working fine. Once I have that, I constructed a convolution neural network model similar to the LeNet model. I thought this model might be appropriate because the LeNet model worked well for the Traffic Sign Clasffier project so that seemed like a good starting point.

I started with just one lap of center lane driving data.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had high mean squared errors (MSE) on both the training set and the validation set. This implied that the model was underfitting. When I used my model to drive the car in the simulator, it would just spin around.

To reduce underfitting, I did the following:

* 1) Add a preprocessing step with normalization and mean centering
This step alone helped me drastically reduce both the training and validation loss.

* 2) Augment the data by flipping the images
This gave me additional training data without using the simulator to generate the data. Both training and validation loss went down for a bit.

* 3) Add more training data
I added one additional lap of center lane driving, plus one lap of recovery driving, and one lap of reverse driving. I made sure to apply the flip image step to all training data.

* 4) Use the left and right images and add a correction factor to the angle
So far I had just been using the center images. Here I utilized the left and right images and added an arbitrary correction factor of 0.2, i.e. +0.2 for the left images and -0.2 for the right images.

After the above mentioned steps, my model could drive the car reasonably well for a bit in the simulator, but would veer off the track when it gets to the first big curve.

Now that I have decent data but the model is still poor, this made me believe that I should focus on improving my network parameters instead. Instead of using LeNet, I decided to apply the Nvidia model, with 5 convolutional layers. This really helped me improve my model and the car was able to drive automously around the track without leaving the road, although there were a few instances the car would stay close to the left side of the road as opposed to the center.

I tried to collect more data using training data from track 2, but it did not help me improve the model, so I stopped there.

####2. Final Model Architecture

The final model architecture (model.py lines 69-111) consisted of a convolution neural network with the following layers and layer sizes:

* 1) CNN: 24 filter, with filter size of 5x5, and strides of 2x2
* 2) CNN: 36 filter, with filter size of 5x5, and strides of 2x2
* 3) CNN: 48 filter, with filter size of 5x5, and strides of 2x2
* 4) CNN: 64 filter, with filter size of 3x3, and strides of 1x1
* 5) CNN: 64 filter, with filter size of 3x3, and strides of 1x1
* 6) Fully connected: 100 nodes
* 7) Fully connected: 50 nodes
* 8) Fully connected: 10 nodes
* 9) Fully connected: single node

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center lane driving][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover after having drifted to the left or right sides of the track. These images show what a recovery looks like starting from right back to center :

![recovery 1][image3]
![recovery 2][image4]
![recovery 3][image5]

Then I repeated this process on track two in order to get more data points. However, data from track two did not help me improve the model, so I did not end up using these additional data.

To augment the data sat, I also flipped images and angles thinking that this would give me more training data points. For example, here is an image that has then been flipped:

![original][image6]
![fliipped][image7]

After the collection process, I had over 30K data points. I then preprocessed this data by applying normalization (divide every pixel by 255) and mean centering (subtract 0.5 from every normalized pixel).

I finally randomly shuffled the data set and put 20% of the data into a validation set.

Since I have so many data points, I decided to use a generator method for better memory management.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the observation that validation loss seemed to stay steady after 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
