# OCRA source code

- For data generation and loading 

    1) stimuli_util.ipynb includes all the codes and the instructions for how to generate the datasets for the two tasks; MultiMNIST and MultiMNIST Cluttered. 
    2) loaddata.py should be updated with the location of the data files for the two tasks if not the default used.

- For training and testing the model:

    1) OCRA_demo.ipynb includes the code for building and training the model. In the first notebook cell, a hyperparameter file should be specified. Two parameter files are provided here (different settings are discussed in the supplementary file)

    2) multimnist_params.txt sets all the hyperparameters for MultiMNIST task with 10 glimpses. 

    3) multimnist_cluttered_params.txt sets all the hyperparameters for MultiMNIST Cluttered task with 5 glimpses. 

    4) This notebook also includes code for testing a trained model and also for plotting the attention windows for sample images. 

# Image-level accuracy averaged from 5 runs

| Task (Model name)                     | Error Rate (SD) |
|---------------------------------------|-----------------|
| MultiMNIST (OCRA-10glimpse)           | 5.08 (0.17)     |
| Cluttered MultiMNIST (OCRA-7glimpse)  | 7.12 (1.05)     |
| MultiSVHN (OCRA-12glimpse)            | 10.07 (0.53)    |

# Trajectory of validation losses across epochs

From MultiMNIST OCRA-10glimpse: 
<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/multimnist-10steps.png" width = 700>

From Cluttered MultiMNIST OCRA-7glimpse
<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/clutter-7steps.png" width = 700>

# Supplementary Results:

<!-- - [Error bars for multiple runs of OCRA](#error-bars-for-multiple-runs-of-OCRA)
  * [MultiMNIST task with 10 glimpses](#multimnist-task-with-10-glimpses)
  * [MultiMNIST Cluttered task with 5 glimpses](#multimnist-cluttered-task-with-5-glimpses) -->
- [Object-centric behavior](#object-centric-behavior)
  * [MultiMNIST Cluttered task with 5 glimpses](#multimnist-cluttered-task-with-5-glimpses-1)
  * [MultiMNIST Cluttered task with 3 glimpses](#multimnist-cluttered-task-with-3-glimpses)
- [The Street View House Numbers Dataset](#the-street-view-house-numbers-dataset)



## Object-centric behavior 

The opportunity to observe the object-centric behavior is bigger in the cluttered task. Two main reasons are as follows: first, the ratio of the glimpse size to the image size is small forcing the model to move and select objects to accurately recognize the objects. Second, the number is allowed very few glimpses (we experimented with 3 and 5) forcing it to make use of its object-centric representation to find the objects without being distracted by the noise segments. 
We have included many more examples of the model behavior with both 3 and 5 glimpses to show this behavior. 


### MultiMNIST Cluttered task with 5 glimpses

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_5_time_steps_t0.gif">

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image00.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_5_time_steps_t1.gif">

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image01.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_5_time_steps_t2.gif">

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image02.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_5_time_steps_mmc.gif">

-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

### MultiMNIST Cluttered task with 3 glimpses 


<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_3_time_steps_t6.gif">

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image06.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_3_time_steps_t7.gif">

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image07.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_3_time_steps_t8.gif">

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image08.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_3_time_steps_mmc.gif">

-------------------------------------------------------------------------------------------------------

## The Street View House Numbers Dataset
This Street View House Numbers (SVHN) dataset includes real-world examples of house numbers. Each image can have from 1 to 5 objects. This tests OCRA on both its applicability to more complex real-world stimuli and also on handling a varying number of objects in an image with duplicates. 

The dataset includes train, test, and extra datasets. We combined the train and extra set to create a bigger training size and also converted the images to grayscale following Ba et al. (2015). 

We made two small changed to the model for this task. First, we increased the number of convolutional filters in the backbone from 32 to 64 in each of the two layers. Second, we added a readout layer to predict the digits in a sequence based on the capsule lengths as the model makes its pass across the image. The resulting model had 5.1 Mil parameters. 

We train the model to "read" the digits from left to right by having the order of the predicted sequence match the ground truth from left to right. We allow the model to make 12 glimpses, with the first two not being constrained and the capsule length from every following two glimpses will be read out for the output digit (e.g. the capsule lengths from the 3rd and 4th glimpses are read out to predict digit number 1; the left-most digit). Below are sample behaviors from our model.

The top five rows show the original images, and the bottom five rows show the reconstructions

![SVHN_gif](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image_grid_SVHN.png)


The generation of sample images across 12 glimpses

![SVHN_gif](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_sequences_SVHN.png)

The generatin in a gif fromat 

![SVHN_gif](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/generated_12_time_steps_SVHN.gif)

The model learns to detect and reconstruct objects. The model achieved ~2.5 percent error rate on recognizing individual digits and ~10 percent error in recognizing whole sequences still lagging SOTA performance on this measure. We believe this to be strongly related to our small two-layer convolutional backbone and we expect to get better results with a deeper one, which we plan to explore next. However, the model shows reasonable attention behavior in performing this task. 

Below shows the model's read and write attention behavior as it reads and reconstructs one image.

<img src="https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/image08_svhn.png">

Herea are a few sample mistakes from our model:

![SVHN_error1](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/im_t3.png)

ground truth  [ 1, 10, 10, 10, 10]

prediction    [ 0, 10, 10, 10, 10]

![SVHN_error2](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/im_t16.png)

ground truth  [ 2,  8, 10, 10, 10]

prediction    [ 2,  9, 10, 10, 10]

![SVHN_error3](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/im_t20.png)

ground truth  [ 1,  2,  9, 10, 10]

prediction    [ 1, 10, 10, 10, 10]

![SVHN_error4](https://raw.githubusercontent.com/Recurrent-Attention-Models/OCRA/main/figures/im_t30.png)

ground truth  [ 5,  1, 10, 10, 10]

prediction    [ 5,  7, 10, 10, 10]
