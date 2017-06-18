#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/output_15_0.png "Data Augmentation"
[image2]: ./examples/output_12_0.png "TrainingSet-Steering angle distribution"
[image3]: ./examples/output_22_3.png "MSE curves"


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
Alternatively, I have also included a Ipython notebook which I have used for training and validation purposes, the code in model.py is just a copy and re-formatting of the code in Ipython notebook.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I did not try to play with model acrchitecture choice too much and hence started with Nvidia model, only made slight changes to it. I added a 1x1 conv layer in starting so that the network can choose its own color channel. However, it should be noted that
in such a case the network woulld require a decent amount of training data. After the initial 1x1 layer I have Conv->Activation->Maxpooling layers replicated thrice followed by falttening and dropout. The last three layers are dense layer of size 100, 100, 10
followed by a single neuron in the output layer. I have used Relu activation eevrywhere except the last neuron output. The model summary table is given below:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_25 (Convolution2D) (None, 64, 96, 1)     4           convolution2d_input_7[0][0]      
____________________________________________________________________________________________________
convolution2d_26 (Convolution2D) (None, 62, 94, 24)    240         convolution2d_25[0][0]           
____________________________________________________________________________________________________
activation_37 (Activation)       (None, 62, 94, 24)    0           convolution2d_26[0][0]           
____________________________________________________________________________________________________
maxpooling2d_19 (MaxPooling2D)   (None, 31, 47, 24)    0           activation_37[0][0]              
____________________________________________________________________________________________________
convolution2d_27 (Convolution2D) (None, 29, 45, 36)    7812        maxpooling2d_19[0][0]            
____________________________________________________________________________________________________
activation_38 (Activation)       (None, 29, 45, 36)    0           convolution2d_27[0][0]           
____________________________________________________________________________________________________
maxpooling2d_20 (MaxPooling2D)   (None, 14, 22, 36)    0           activation_38[0][0]              
____________________________________________________________________________________________________
convolution2d_28 (Convolution2D) (None, 12, 20, 48)    15600       maxpooling2d_20[0][0]            
____________________________________________________________________________________________________
activation_39 (Activation)       (None, 12, 20, 48)    0           convolution2d_28[0][0]           
____________________________________________________________________________________________________
maxpooling2d_21 (MaxPooling2D)   (None, 6, 10, 48)     0           activation_39[0][0]              
____________________________________________________________________________________________________
flatten_7 (Flatten)              (None, 2880)          0           maxpooling2d_21[0][0]            
____________________________________________________________________________________________________
dropout_7 (Dropout)              (None, 2880)          0           flatten_7[0][0]                  
____________________________________________________________________________________________________
dense_25 (Dense)                 (None, 100)           288100      dropout_7[0][0]                  
____________________________________________________________________________________________________
activation_40 (Activation)       (None, 100)           0           dense_25[0][0]                   
____________________________________________________________________________________________________
dense_26 (Dense)                 (None, 100)           10100       activation_40[0][0]              
____________________________________________________________________________________________________
activation_41 (Activation)       (None, 100)           0           dense_26[0][0]                   
____________________________________________________________________________________________________
dense_27 (Dense)                 (None, 10)            1010        activation_41[0][0]              
____________________________________________________________________________________________________
activation_42 (Activation)       (None, 10)            0           dense_27[0][0]                   
____________________________________________________________________________________________________
dense_28 (Dense)                 (None, 1)             11          activation_42[0][0]              
====================================================================================================
Total params: 322,877
Trainable params: 322,877
Non-trainable params: 0

The code for the model is given in code cell 16 of Ipython notebook (`build_nvidia_model` function). Alternately, the same function is also present in model.py (line 142-180). 


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting as stated above. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (see function `build_nvidia_model` in model.py).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the nvidia model. I thought this model might be appropriate because it has been used for a similar task by Nividia. I had made some small modification to this architecture as mentioned in `Model Architecture and Training Strategy` above.
This is by far the most difficult assignment, so I had to spent a lot of time in understanding what is going on with the model. After going through the blog post and discussion of some fellow students, I decide to adopt the following strategy:
1. Fix the model architecture similar to Nvidia model except added a 1x1 conv layer in front so that the network can choose it's own channel, Further incorporated drop out to reduce overfitting.
2. In order to reduce the training and resource requirements, in addition to removing the unwanted image portion(like trees, sky), I re-sized the image to 64x96.
3. Training Data set preparation: This was an iterative step, I started with Udacity training set but it wasn't sufficient, so I added more training images for the track location where the network was failing, added some recovery data , adding more data for the dirt after the bridge and so on.
4. Further I tried to create a balanced dataset, since most of time the car was moving straight and the network would get biased towards moving straight, so I used the left and right camera images to create a more balanced training set. The code is present in code cell 36 and 40 in Ipython notebook (line 290-330 in model.py)
5. While training, I performed random horizontal and vertical shifts, random shadows (though this was not essential, the network worked just fine without this step but still added it so that it get generalized better), added random brightness and flips (see function `generator_data`). I also added small amount of multiplicative noise
to the steering angle (uniformly distributed between normalized anngle of +/-0.1) to account for incorrect steering angle due to driving behaviour on the track.

In order to gauge how well the model was working, I split my data into training set and test set. No random augmentation (flips/horizontal/vertical/brightness/shadowing) were performed on test set. At the end of training, both the traing error and test error were plotted. Note that I have trained the model incrementally (re-loaded the previous weights and re-trained the network again after adding more recovery data wherever required)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is mentioned in ###1 above (see the function `build_nvidia_model` in model.py lines 142-180) 

####3. Creation of the Training Set & Training Process

This set is explained in `Solution Design Approach` above. I present some of the sampled images showing the data augmentation employed while training the network.


The image below shows the data augmentation used while training the network
![alt text][image1]

The figure below shows the distribution of the steering angle in the training data set.
![alt text][image2]

The figure below shows the mean sqaure error curves for the training and the test set
![alt text][image3]

