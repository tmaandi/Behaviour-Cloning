# Behavioral Cloning Project


The goals of this project were:

    Use the simulator to collect data of good driving behavior
    Build a convolution neural network in Keras that predicts steering angles from images
    Train and validate the model with a training and validation set
    Test that the model successfully drives around track one without leaving the road


My project includes the following files:

    model.py containing the script to create and train the model
    drive.py for driving the car in autonomous mode
    model.h5 containing a trained convolution neural network
    image_process_demo.py for outputting samples of processed images
    writeup_report.md and writeup_report.ipynb summarizing the results

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

    python drive.py model.h5

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

### Architecture

My model consists of a convolutional neural network similar to NVIDIA's CNN used for training and driving a car autonomously. This network uses 5 x 5 and 3x3 filter sizes and depths between 24 and 64 (refer model.py). The network implementation is done using Keras library. A few minor differences like input layer size, cropping, normalization, an additional Dropout layer and convolutional filter stride size exist between my network and this standard architecture shown below.

[nvidia]: ./examples/cnn-architecture-nvidia.png "NVIDIA CNN Architecture [Source: NVIDIA DevBlog]"
![alt text][nvidia]


The model includes RELU activations in the convolutional layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer. Another keras lambda layer is used to crop the image data. The benefit of using cropping layer as a part of the network is so that it can use the GPU capability to crop multiple images simultaneously thus saving considerable time in training.
The model also contains a dropout layer with a dropout rate of 40% in order to reduce overfitting.
Below is the image which shows how the model was 'overfitting' and was tackled by reducing the number of epochs and adding a dropout layer with a 40% dropout rate. I inititally added three dropout layers but it hurt the training loss and validation accuracy and I settled for one layer.  

[overfitting]: ./examples/Loss_and_Validation_overfitting.png "Network Overfitting during training"
![alt text][overfitting]
[not_overfitting]: ./examples/Loss_and_Validation.png "Network Performance with fewer epochs and Dropout layer"
![alt text][not_overfitting]

The data normalization accompanied with mean-zeroing was done as a Keras lambda layer. No more modifications were required by the network architecture for doing the task.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### Training Strategy
[//]: # (Image References)

[left_right_center]: ./examples/Left_Right_and_Center.png "Left, Right and Center"
[Flipped]: ./examples/Flipped.png "Flipping Effect"

The model was trained and validated on different data sets to ensure that the model was not overfitting (used test_train_split function for splitting training data). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
Training data was chosen to keep the vehicle driving on the road. The data provided for one complete lap was used as the primary data source. I used a combination of centre, left and right camera data. Using left and right camera data helped in cornering through sharp turns. The steering angle value was corrected accordingly for left and right camera training data. I observed that using too high steering correction factor would make car zig-zag all over the place and using too low steering correction would hurt the training and the car would perform even worse than center lane only training. The image below shows the view from three different cameras:
![alt text][left_right_center]

In addition, I also flipped the centre camera data and used it for training. This helped the network generalize better and I didn't have to use the clockwise lap traverse data as "np.flipr" had the similar effect on the data as shown below:

![alt text][Flipped]

The video video_result.mp4 of the car doing a full lap in the simulator can be found in the directory 