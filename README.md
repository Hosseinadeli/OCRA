# OCRA (Object-Centric Recurrent Attention) source code

[Hossein Adeli](https://hosseinadeli.github.io/) and [Seoyoung Ahn](https://ahnchive.github.io/)

Please cite this article if you find this repository useful:

[Recurrent Attention Models with Object-centric Capsule Representation for Multi-object Recognition](https://arxiv.org/abs/2110.04954)<br/>
H Adeli, S Ahn, G Zelinsky - arXiv preprint arXiv:2110.04954, 2021 - arxiv.org <br/>

[[pdf](https://arxiv.org/pdf/2110.04954.pdf/)]

[[google scholar](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=EdIFZpQAAAAJ&sortby=pubdate&citation_for_view=EdIFZpQAAAAJ:mVmsd5A6BfQC/)]
-------------------------------------------------------------------------------------------------------
- For data generation and loading 

    1) stimuli_util.ipynb includes all the codes and the instructions for how to generate the datasets for the three tasks; MultiMNIST, MultiMNIST Cluttered and MultiSVHN. 
    2) loaddata.py should be updated with the location of the data files for the tasks if not the default used.

- For training and testing the model:

    1) OCRA_demo.ipynb includes the code for building and training the model. In the first notebook cell, a hyperparameter file should be specified. Parameter files are provided here (different settings are discussed in the supplementary file)

    2) multimnist_params_10glimpse.txt and multimnist_params_3glimpse.txt set all the hyperparameters for MultiMNIST task with 10 and 3 glimpses, respectively. 

    OCRA_demo-MultiMNIST_3glimpse_training.ipynb shows how to load a parameter file and train the model. 

    3) multimnist_cluttered_params_7glimpse.txt and multimnist_cluttered_params_5glimpse.txt set all the hyperparameters for MultiMNIST Cluttered task with 7 and 5 glimpses, respectively. 

    4) multisvhn_params.txt sets all the hyperparameters for the MultiSVHN task with 12 glimpses. 
    
    5) This notebook also includes code for testing a trained model and also for plotting the attention windows for sample images. 
    
    OCRA_demo-cluttered_5steps_loadtrained.ipynb shows how to load a trained model and test it on the test dataset. Example pretrained models are included in the repository under pretrained folder. [Download](https://drive.google.com/drive/folders/1lBdcMmCdDjumpAm2wlex-PAJm5Wwxtsz?usp=sharing) all the pretrained models. 
    
    

# Image-level accuracy averaged from 5 runs

| Task (Model name)                     | Error Rate (SD) |
|---------------------------------------|-----------------|
| MultiMNIST (OCRA-10glimpse)           | 5.08 (0.17)     |
| Cluttered MultiMNIST (OCRA-7glimpse)  | 7.12 (1.05)     |
| MultiSVHN (OCRA-12glimpse)            | 10.07 (0.53)    |

# Validation losses during training

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

The opportunity to observe the object-centric behavior is bigger in the cluttered task. Since the ratio of the glimpse size to the image size is small (covering less than 4 percent of the image), the model needs to optimally move and select the objects to accurately recognize them. Also reducing the number of glimpses has a similar effect, (we experimented with 3 and 5) forcing the model to leverage its object-centric representation to find the objects without being distracted by the noise segments. 
We include many more examples of the model behavior with both 3 and 5 glimpses to show this behavior. 


### MultiMNIST Cluttered task with 5 glimpses

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_5_time_steps_t0.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image00.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_5_time_steps_t1.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image01.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_5_time_steps_t2.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image02.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_5_time_steps_mmc.gif">

-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------

### MultiMNIST Cluttered task with 3 glimpses 


<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_3_time_steps_t6.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image06.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_3_time_steps_t7.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image07.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_3_time_steps_t8.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image08.png">

-------------------------------------------------------------------------------------------------------

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_3_time_steps_mmc.gif">

-------------------------------------------------------------------------------------------------------

## The Street View House Numbers Dataset

We train the model to "read" the digits from left to right by having the order of the predicted sequence match the ground truth from left to right. We allow the model to make 12 glimpses, with the first two not being constrained and the capsule length from every following two glimpses will be read out for the output digit (e.g. the capsule lengths from the 3rd and 4th glimpses are read out to predict digit number 1; the left-most digit and so on). Below are sample behaviors from our model.

The top five rows show the original images, and the bottom five rows show the reconstructions

![SVHN_gif](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image_grid_SVHN.png)


The generation of sample images across 12 glimpses

![SVHN_gif](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_sequences_SVHN.png)

The generatin in a gif fromat 

![SVHN_gif](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/generated_12_time_steps_SVHN.gif)

The model learns to detect and reconstruct objects. The model achieved ~2.5 percent error rate on recognizing individual digits and ~10 percent error in recognizing whole sequences still lagging SOTA performance on this measure. We believe this to be strongly related to our small two-layer convolutional backbone and we expect to get better results with a deeper one, which we plan to explore next. However, the model shows reasonable attention behavior in performing this task. 

Below shows the model's read and write attention behavior as it reads and reconstructs one image.

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/image08_svhn.png">

Herea are a few sample mistakes from our model:

![SVHN_error1](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/im_t3.png)<br/>
ground truth  [ 1, 10, 10, 10, 10]<br/>
prediction    [ 0, 10, 10, 10, 10]

![SVHN_error2](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/im_t16.png)<br/>
ground truth  [ 2,  8, 10, 10, 10]<br/>
prediction    [ 2,  9, 10, 10, 10]

![SVHN_error3](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/im_t20.png)<br/>
ground truth  [ 1,  2,  9, 10, 10]<br/>
prediction    [ 1, 10, 10, 10, 10]

![SVHN_error4](https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/im_t30.png)<br/>
ground truth  [ 5,  1, 10, 10, 10]<br/>
prediction    [ 5,  7, 10, 10, 10]

-------------------------------------------------------------------------------------------------------
Some MNIST cluttered results 

Testing the model on MNIST cluttered dataset with three time steps

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/mc_grid.png">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/mc_1.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/mc_2.gif">

<img src="https://raw.githubusercontent.com/hosseinadeli/OCRA/main/figures/mc_3.gif">

-------------------------------------------------------------------------------------------------------
Code references:

1) [XifengGuo/CapsNet-Pytorch](https://github.com/XifengGuo/CapsNet-Pytorch) <br/>
2) [kamenbliznashki/generative_models](https://github.com/kamenbliznashki/generative_models/blob/master/draw.py) <br/>
3) [pitsios-s/SVHN](https://github.com/pitsios-s/SVHN-Thesis/blob/master/src/multi_digit/svhn.py) <br/>
