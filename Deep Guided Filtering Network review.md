## FAST END-TO-END TRAINABLE GUIDED FILTER
### REVIEW

#### Introduction

This is a short review of the paper "Fast End-to-End Trainable Guided Filter" written by Huikai Wu et. al., that discusses as the name suggests an interesting method of using deep learning techniques to perform image processing techniques as well as computer vision tasks not only accurately but also fast using end to end training. In this article I attempt to explain as simply as possible and to the best of my understanding, the concept behind this paper and how it can be implemented. Along with this article, I have converted the [original repository](https://github.com/wuhuikai/DeepGuidedFilter) which is in pytorch into a [keras(tensorflow backend) notebook](https://github.com/shreyasrajesh) for one of the image processing tasks(image retouching) to provide much better understanding of the code and the underlying algorithms as well.

This is a fairly mathematical paper and although not very complicated, it contains a lot of math due to the variety of the algorithms being used in this paper. My attempt is to simplify this to the best of my abilities. I shall go about this review by first giving an eagle's eye view about the paper, then delve into the each of the individual algorithms and finally talk about my attempt to execute this paper. I don't intend to discuss the results in detail, just enough to explain the merits of this paper. 

#### Purpose

Image processing and computer vision have recently taken massive strides harnessing the capabilities of deep learning especially, convolutional neural networks. However, despite these techniques being really accurate the biggest problem remains that these processes are particularly slow on high resolution images. The only workaround until now was to convert the high resolution images to low resolution images and then back using joint upsampling. However, lack of good joint upsampling methods as result in a loss of accuracy in a bid to obtain performance gains.

This paper proposes a fully differentiable guided filter layer that can be integrated and trained along with the CNN and perform an efficient and accurate joint upsampling. It clearly outdoes the bilateral filter that was being used for joint upsampling as it not only provides more accurate results, it also generalizes well to various tasks by generating task-oriented guidance maps. This end-to-end trainable guided filtering layer along with a learning guidance map provides improvement on the state-of-the-art results on multiple image processing tasks and runs 10-100x faster at the same time.

#### Problem Statement

None of the methods thus far have used a fully differentiable block that can be trained along with the network. Most methods use a bilateral filter which requires massive amount of computational resources and works as a post processing operation creating a bottleneck in the process.

This paper uses the state of the art CNN, [Context Aggregation Networks](https://arxiv.org/pdf/1709.00643.pdf) for a variety accelerated image processing operations and builds the trainable guided filter layer on top of it to obtain the published results.

Given an image $I_h$ and the corresponding low resolution output $O_l$ the objective is to obtain $O_h$ which is visually similar to $O_l$ but retains the sharp features of $I_h$ like the edges and details. This is done by adopting a fast joint upsampling solver(He _et al._) and formulate it as a trainable filter module. The framework for the entire trainable filter network proposed is as shown below.
![alt text](images/Screenshot\ from\ 2018-06-23\ 20-13-44.png)
This clearly shows that this can be split into the deep guided filtering network(DGFN) and the Context Aggregation Network(CAN).
The guided filtering layer block is explained in much greater detail in the next section. Finally, I personally stuck to a single image processing task for simplicity of explanation and corresponding code throughout the tutorial. The task I have considered is Image Retouching(Automatic Photoshop). Most of the concepts and explanation however, is identical to this tutorial. There are only a few minor changes required in the code.

#### Deep Guided Filtering Network
The core of this paper lies in this module. The image is the computational graph of the guided filtering layer giving an overview of the various tasks being performed.  
![alt text](images/Screenshot\ from\ 2018-06-23\ 20-14-08.png)
The first and most important observation is that $O_h$ can be backpropogated to all the inputs providing direct supervision of the high resolution outputs on the inputs also generating a better $O_l$ for the guided filtering layer to restore. The first block in this computational graph is, 
##### 1. Transformation Function$(F(I))$

This block as the name suggests is a trainable transformation function containing two convolutional layers and transform the inputs, $I_h$ and $I_l$ with any channel size into ones with same channel size as the output and represented by $G_h$ and $G_l$ respectively. This fully convolutional neural network consists of only two convolutional layers between which are an adaptive normalisation and a leaky ReLU layer. Both convolutions are pointwise convolutions with the first one having a fixed channel size of 64 as the only task being performed here is channel size modification and operations along only that axis are necessary. (Each conv. is esentially just a linear operation along the channel axis on each pixel)

##### 2. Mean Filter$(f_{\mu})$

This filter is just a simple box filter which is essentially a linear filter that averages out the values of itself and all its neighbouring pixels upto a certain distance given by radius, r(hyperparameter). This is done on the transformed low resolution input as well as the low resolution output and the results are fed into the local linear model. 

##### 3. Local Linear Model

As the name suggests this is a simple block that converts the mean filter outputs($\bar G_l, \bar O_l$) into the required summations and coefficients for the next bilinear upsampling block. This block generates two important variables $A_l$ and $b_l$, which are obtained by a series of mathematical summations and products. These operations also include a regularization term($\epsilon$) which is a hyperparameter and set to $10^{-08}$ for our purposes. 

##### 4. Bilinear Upsample($f_{\uparrow}$)

This block upsamples the coefficients it receives from the local linear model($A_l$ and  $b_l$) using the inbuilt bilinear upsampling operator in kears/pytorch. In keras it is written as a convolutional layer but essentially does the same task.

##### 5. Linear Layer

This is the final layer and is a basic single layer weight multiplication type operation(wx+b operation as I call it) and uses the bilinearly upsampled coefficients on the high resolution input image to generate the final high resolution output image.

As it can be clearly observed, each of these operations are entirely continuous and differentiable thus making it an end-to-end trainable filter.
Now, that the entire computational graph is briefly described, we consider the next block that is the convolutional block which is integrated along with the rest of the guided filtering layer in a coarse to fine manner. 

##### Context Aggregation Network(CAN)

The Context Aggregation Network represented as $C_l(I_l)$, is applied after downsampling the low resolution image to obtain the low resolution output($O_l$). This usage of the Context Aggregation Network at a much smaller resolution drastically speeds it up as expected, improving not only the computational resources required but also the overall speed of the entire process. The architecture of this network can be varied according to the charecteristics of different tasks to obtain optimal results however, Context Aggregation Networks as proposed by [Chen _et al._](https://arxiv.org/pdf/1709.00643.pdf)  was used as the base network for all the image porcessing tasks. The networks used for the Computer Vision tasks(not being considered here) were however different and independent to that task.

Finally, the loss function used to train the entire model is an $l2$ loss with a pixelwise comparison of the generated high resolution output to the target output as annotated by experts. 


#### Execution

This section briefly desribes my attempts and experiments with the implementation of this project. First off, my aim was to convert the already existing [github repository](https://github.com/wuhuikai/DeepGuidedFilter) into a much simpler easier to understand IPython Notebook written in keras. This would not only help everyone like me who is attempting to understand the code behind this paper but also provide me with a clear end to end understanding of the code. However, of the repository I have only considered only one of the image processing tasks as the code required for the other tasks are very similar and it would increase the clutter making it much harder to understand and follow which is the primitive aim. 

On scouring through the code in the repository I noticed as there are a significant number of independent python files with dependencies on each other, it would be wise to choose the one with no dependencies and slowly work my way outward finally ending with the one with the most dependencies. This is mainly because IPython notebooks follow a chronological order and anything above a new piece of code can be referred to without any extra requirements. Hence, this approach of tackling the code was chosen. After making this decision I embarked on the journey to convert all the code from pytorch to keras. I started with the mean filter and guided filter and worked my way outwards to the deep guided filter with the upsampling and finally added in the Context Aggregation Networks and the task specific guidance map$(F(I))$ with all training related code in the end. 

Converting pytorch to keras code is not as easy as it first seems. Pytorch is written to provide more support towards custom layers and non-standard layers and hence a clear choice by the author. However, keras is not the best choice when it comes to codes requiring a lot of custom layers and tensor operations. This took longer than I had originally expected, however, the keras backend provides all the required support for tensor operations and all custom layers. I learnt a lot about the source code of both keras and pytorch in the process as a pleasent surprise. 

I faced a few hiccups during this process which I believe were the points of greatest learning throughout this process. I would like to discuss a couple of these here for knowledge and to avoid repitition in the future.

##### 1. Code transfer

The first and biggest mistake I made in this endeavour was that I tried to convert the pytorch code line for line into keras. I took for granted that as both pytorch and keras are powerful deep learning frameworks that will have a sort of 1 to 1 mapping in terms of syntax and capabilities. I hence ended up spending a good amount of time in simply looking for stuff that never existed in the keras documentation. Beyond a point however, I rectified this line of thinking and started looking at entire classes as a block of code and tried converting these blocks as a whole into keras even if it meant the block looking entirely different at a linewise level from the pytorch code. 

##### 2. Upsampling layer

This is still an unsolved issue. The pytorch framework allows a simple upsampling with a mode flag to choose the type of upsampling. However, keras's only upsampling support is a predefined convolutional layer which doesn't seem to quite do the same operation as the pytorch tensor operation and doesn't have the flexibility required in this case. I have for now used the keras convolutional layer as opposed to writing my own entirely new module using backend tensor operations as I look for something similar to pytorch in the keras docs, however the doing it from scratch seems to be the more accurate option. 

##### 3. Model Conversion

This continues to remain an issue, although I have been looking for solutions to from the day I started the project. I have looked at several platforms, scripts and even entire repositories to convert pretrained pytorch(.pth) models into corresponding pretrained keras(.hdf5 or .h5) model files. So far no option has come through each one not viable for various different reasons. The main and most common reason for the failure of most of the models is the galore of custom layers which these codes are unequipped to handle. I begin spending some time and trying to write my own python script to implement this conversion but as amateur coder with not much support on the subject online I struggled too much and hence dropped the idea. I hope to eventually spend enough time and come up with this script or atleast find one that can be used in this case and implement it successfully.
 
 ##### 4. Dataset
 
This is a slightly smaller issue however, worthy of a mention for the amount of time I spent on it. The original project repo is trained on the MIT FiveK Adobe dataset that is 50GB and hence a pain to deal with. The code in the original repository has provided a few scripts for the data preprocessing to prime it for the ensuing training and testing. However, these scripts don't seem to do the job for reasons I am not able to debug. Therefore, I first chose a small part of the dataset as the dataset is massive and very heavy to operate on. Following this, I have written spent a fair amount of time writing my own scripts for the same with the ones provided in the repository as a base and for reference whenever required. The IPython Notebook itself doesn't contain this code and was left out for readability, brevity and to avoid complexity. However, this code is very similar to the code in the original repository and that can be referred to.
The massive size of not only the data but also the annotations for each image is an issue for the IPython Notebook and hence a very small set has been considered just for demonstration purposes. 

#### Conclusion

After spending a considerable amount of time on this project that started a seemingly fairly simple task, I have come to learn a great deal regarding at each pitfall throughout the process. I hope to complete the entire task and fix all the issues I have had during execution as soon as possible and further move on to optimize the IPython Notebook and this review to the best possible extent and share my efforts on a public forum(ex. Medium) along with a blog post derived mainly from this review. I believe that is still a lot of scope for improvement with this project and the deadline restrictions have led to incomplete justification and I do not intend to let it remain this way. 

I believe this work is still fairly new and its potential applications are yet to be realised. Further despite the user friendly open source code provided by the author the datasets are hard to find and the multiple tasks coded in make the code seem a bit cluttered. I hope to eliminate these issues with my notebook version helping people use it more easily and modify it for their specific purposes with almost no struggle. I will strive hard and am determined to see this project through as I have a very clear picture in my head as to how the end product needs to be delivered but lack the execution currently and want to realise this image of the completed project.  