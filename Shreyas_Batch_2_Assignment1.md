# EPOCHS
This short article deals with the meaning and significance of epochs in training in deep learning. 
An *epoch* can be defined as one complete pass through the entire training dataset. An epoch consists of multiple batches, each batch consisting of a small part of the training data. The batch size is chosen based on the GPU speed requirements and memory availability however, the epoch size depends entirely on the size of the dataset. The time taken to complete one Epoch depends on the batch size and accordingly the GPU capacity. 
An iteration is often confused with an epoch. They mean the same thing only if all the parameters are backpropogated and updated once it has passed through the entire dataset. However, every time the parameters are updated irrespective of how much of the training dataset the model has seen it counts as one iteration. So, if there are batches and the parameters are updated every once in a few batches it is possible that there are multiple iterations within a single epoch.
#### 1 epoch = N iterations = (training dataset)/(batch size)
An interesting article explaining this better and in more detail is given below:
[Epoch vs Batch Size vs Iterations – Towards Data Science](https://towardsdatascience.com/epoch-vs-iterations-vs-batch-size-4dfb9c7ce9c9)
<br>
# ACTIVATION FUNCTIONS
Activation functions are essentially a functional mapping between the input and the output. This means that it converts the input after passing through the neural network into meaningful output. If no activation function is applied the output will be a simple linear function and the complexity of the prolem cannot be captured and the model will not be able to learn any of the required inputs(e.g. images, speech, text). So these activation functions provide the true power to the neural network models and help them learn whatever input is being provided. Another important note about activation functions is that they need to be differentiable as they need to be backpropogated through as well. The three most popular activation functions are:
1. Sigmoid Function
![alt text](https://cdn-images-1.medium.com/max/800/0*WYB0K0zk1MiIB6xp.png =400x)
2. Tanh - Hyperbolic Function
![alt text](https://cdn-images-1.medium.com/max/800/0*VHhGS4NwibecRjIa.png =400x)
3. ReLU - Rectified Linear Units
![alt text](https://cdn-images-1.medium.com/max/800/0*qtfLu9rmtNullrVC.png =500x)

Rectified Linear Units are further divided into multiple sub categories and a host of other activation functions are used as well, however these are the most commonly used 3 activation functions.
Further information on activation functions can be obtained at the following link:
[Activation Functions: Neural Networks – Towards Data Science](https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6)
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE2MTk0NzU0NjFdfQ==
-->
