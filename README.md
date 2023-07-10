
# Self-Driving Car Steering Angle Prediction Based on Image Recognition

# Description:

Tesla,Inc lead by Elon Musk has created revolution in the autonomous driving in the recent years. This opened up a plethora of reseaarch points in autonomous auto industry. One such topic is Intrusion Detection Systemv(IDS) for autonomous vehicles. This plays an important role in the anatomy of autonomous vehicles to develop cognigance for detecting trusted sensors and extract information from them to act upon actutators. The following project is a part of test bed being developed for my research.

# Problem Statement:

A self-driving car is loaded with cameras which capture images of the road in front of it. Using these images in realtime the model should be able to analyze the direction of the road and predict how much the steering wheel must be turned to follow it. For this project, we have a data set of pre-captured images for which we will predict the angle.

The project is an implementation of the paper [Self-Driving Car Steering Angle Prediction Based on Image Recognition] (http://cs231n.stanford.edu/reports/2017/pdfs/626.pdf)

# Dataset:

In this project Udacity driving simulator has been used for generating the dataset.

The driving simulator collects images from three front-facing "cameras," along with the various auxillary information such as throttle, speed, and steering angle. We'll feed camera data into the model and predict steering angles in the [-1, 1] range.

# Approach:

The approach followed in the project is to build a hybrid model using Transfer learning from a pretrained model. 

 The idea of transfer learning is that features learned in the lower layers of the model are likely transferable to another dataset. Of the pre-trained models available, ResNet50 was choosen for its excellent benchmarks . This model was trained on ImageNet. The weights of the first 15 ResNet blocks were
blocked from updating (first 45 individual layers out of 175 total). The output of ResNet50 was connected to a stack of fully connected layers containing 100, 50, 10, and 1 different units respectively. 


The architecture of this model can be seen in figure below with overall parameters being 26,870,183. Each dense layer has elu as activation function. The idead begind using elu is that it fits well with the angle distribution as well as generate values that can be positive or negative. The
model output is a value of angle between -1 and 1.


<img width="412" alt="Screen Shot 2022-05-20 at 8 16 45 PM" src="https://user-images.githubusercontent.com/102194740/169628439-b37ccc09-178c-4620-89de-c0b0d8ec16a3.png">

Figure: Architecture of hybrid model


The input for this model is of shape 100x100x3. All the images from the dataset are cropped to remove unnceccessary features and pre-processed by converting the RGB images to Glaussian Blur. this is done due to the size restriction of resnet50. The data is split intoo 70:30 for training and validation.

<img width="876" alt="Screen Shot 2022-05-20 at 8 42 07 PM" src="https://user-images.githubusercontent.com/102194740/169628636-141d557d-3905-40f6-9e60-abd30e3fbed3.png">

Figure: Comparing the original and pre-processed image



# Results:

The model ws trained using 70:30 split for validation and testing. The model clocked an accuracy of 67.79 % for 100 epochs. The accuracy seems decent enough given the amount of images for training is small. The training loss vs validation loss graph shownn in the figure below depicts a decent training from the algorithm.

<img width="368" alt="Screen Shot 2022-05-20 at 9 19 52 PM" src="https://user-images.githubusercontent.com/102194740/169628937-a2d34f44-6970-4085-8f70-b3f96e3fab88.png">

Figure: Training vs Validation loss from the model


The model is then tested on a [dataset] (https://github.com/SullyChen/Autopilot-TensorFlow) from SullyChen. The results is shown below:

https://user-images.githubusercontent.com/102194740/169629007-63247483-aadf-49be-815e-af388295e74f.mp4

Video: Results of model on SullyChen Auto Pilot Dataset

As shown above the model worked pretty well predecting the steering angle for Auto Pilot Dataset which it never saw during training.




# Run the project:

The ipynb file can be run on jupyter notebook or Colab. Colab is preferred as it reduces the efffore of installation. 

Note: Change the run-time to gpu for faster training of the model


# References:



1. http://cs231n.stanford.edu/reports/2017/pdfs/626.pdf
2.https://github.com/SullyChen/Autopilot-TensorFlow

3. https://github.com/abhinavsagar/self-driving-car

4. https://arxiv.org/pdf/1604.07316.pdf

5. https://rmmody.github.io/pdf/682Project.pdf









