# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./markdown_image/center_lane_driving.jpg "Center Lane Driving"
[image2]: ./markdown_image/recover_from_left.jpg "Recover From Left"
[image3]: ./markdown_image/recover_from_right.jpg "Recover From Right"
[image4]: ./markdown_image/left_camera.jpg "Left Camera"
[image5]: ./markdown_image/middle_camera.jpg "Middle Camera"
[image6]: ./markdown_image/right_camera.jpg "Right Camera"
[image7]: ./markdown_image/before_flip.jpg "Before Flip"
[image8]: ./markdown_image/after_flip.jpg "After Flip"
[image9]: ./markdown_image/before_crop.jpg "Before Crop"
[image10]: ./markdown_image/after_crop.jpg "Before Crop"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 is a video recording of your vehicle driving autonomously at least one lap around the track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with three 5x5 and two 3x3 filter sizes and depths between 24 and 64 (model.py lines 81-96) 

The model includes RELU layers to introduce nonlinearity (code line 86-96), and the data is normalized in the model using a Keras lambda layer (code line 84). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 93). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 101-105). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

* The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 100).
* batch_size = 32
* Number of epoches = 5
* Angle correction = 0.2

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving counter-clockwise.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was having a good starting point, and improve basd on output by iteratively experimenting.

My first step was to use a convolution neural network model similar to the one presented in section 15 (Even more powerful network). I thought this model might be appropriate because in the video it gives relatively good output as starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (model.py line 16). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by adding dropout with keep_prob=0.5 after first fully connected layer, so mean squared error on both sets decreased to an acceptable level.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 81-96) consisted of a convolution neural network with the following visualization.(note: visualizing the architecture is optional according to the project rubric)

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 80, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824      
__5x5, stride 2x2, VALID_________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636     
__5x5, stride 2x2, VALID_________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248     
__5x5, stride 2x2, VALID_________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712     
__3x3, stride 1x1, VALID_________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928     
__3x3, stride 1x1, VALID_________________________________________
flatten_1 (Flatten)          (None, 6336)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700    
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
```

#### 3. Creation of the Training Set & Training Process

I used the provided sample dataset as I looked through all the images in the set. I found out it contains several laps of track 1, including strategies such as center lane driving, recovering from the left side and right sides of the road back to center, and driving in opposite direction. It covers all approaches I can come up with, so I do not bother creating my own data.

Here is an example image of center lane driving:

![alt text][image1]

These images show recovering from the left side and right side of the road:

![alt text][image2]
![alt text][image3]

I also take advantage of the multiple cameras on the car, so center, left, and right images are added to dataset.
Here is an example of 3 angles for one timestamp:

![alt text][image4]
![alt text][image5]
![alt text][image6]

To augment the data set, I also flipped images and angles. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

After the collection process, I had 48,216 data points. I preprocessed them by first cropping 60 pixels from top and 20 pixels from bottom of each image, as those parts of image are not useful in decision-making of the model. Here is an example of before and after cropping:

![alt text][image9]
![alt text][image10]

After cropping I did normalization in the model using a Keras lambda layer (model.py line 84). 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 5 as there is minor improvement after 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
