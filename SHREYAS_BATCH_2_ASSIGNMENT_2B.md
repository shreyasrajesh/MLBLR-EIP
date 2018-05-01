# NEURAL NETWORK MATH
Note: Part A of the assignment can be found at the following [link](https://github.com/shreyasrajesh/MLBLR-EIP).
## 1. FORWARD PASS
Random weights and biases generated from the python script. The link to the python script is [random_number_generator.py](https://github.com/shreyasrajesh/MLBLR-EIP/blob/master/random_number_generator.py). The values output by the script has been used below:
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605| | | | | | |0.0151|0.7409| | |1| |
|1|0|1|1|0.5168|0.2988|0.7560| ||| | | | | | |0.6331| | | |1| |
|0|1|0|1|0.5425|0.9971|0.0385| ||| | | | | | |0.3974| | | |0| |
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
2. Calculation of the hidden layer input and the corresponding hidden layer activation (non-linear transformation) values. This is done using the formulas below:
```
  hidden_layer_input = matrix_dot_product(X,wh) + bh
  hidden_layer_activations = sigmoid(hidden_layer_input)
```
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|0.0151|0.7409| | |1| |
|1|0|1|1|0.5168|0.2988|0.7560| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.6331| | | |1| |
|0|1|0|1|0.5425|0.9971|0.0385| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.3974| | | |0| |
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
3. Repeat step 2 on the hidden layer activation values to get the output layer values. 
```
  output_layer_input = matrix_dot_product(hidden_layer_activations,wout) + bout
  output = sigmoid(output_layer_input)
```
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|0.0151| 0.7409|1.6269|0.8357|1| |
|1|0|1|1|0.5168|0.2988|0.7560| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.6331| |1.7040|0.8461|1| |
|0|1|0|1|0.5425|0.9971|0.0385| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.3974| |1.6052|0.8327|0| |
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
4. The error is then calculated as the difference between the expected output and obtained output.
```
  E = y - output
```
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|0.0151| 0.7409|1.6269|0.8357|1|0.1643|
|1|0|1|1|0.5168|0.2988|0.7560| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.6331| |1.7040|0.8461|1|0.1539|
|0|1|0|1|0.5425|0.9971|0.0385| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.3974| |1.6052|0.8327|0|-0.8327|
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
## 2. BACKPROPOGATION
5. Compute the slope at the output and hidden layer as follows where derivative of sigmoid is given as,
```
derivative(sigmoid) = sigmoid(1-sigmoid)
```
```
Slope of output = derivative(sigmoid(output))
Slope of hidden_layer_input = derivative(sigmoid(hidden_layer_activation))
```
#### Values
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|0.0151| 0.7409|1.6269|0.8357|1|0.1643|
|1|0|1|1|0.5168|0.2988|0.7560| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.6331| |1.7040|0.8461|1|0.1539|
|0|1|0|1|0.5425|0.9971|0.0385| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.3974| |1.6052|0.8327|0|-0.8327|
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
#### Derivatives
|Slope hidden layer|||Slope output|
|:---:|---|---|:---:|
|0.0928|0.1355|0.1205|0.1373| 
|0.0531|0.0688|0.0795|0.1302|
|0.1081|0.1736|0.0861|0.1393|

6. Compute the delta of the output layer and the hidden layer using the newly computed slopes
```
d_output = E * slope_output_layer
d_hidden_layer = (matrix_dot_product(d_output, wout.Transpose)) * slope_hidden_layer
```
#### Values
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|0.0151| 0.7409|1.6269|0.8357|1|0.1643|
|1|0|1|1|0.5168|0.2988|0.7560| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.6331| |1.7040|0.8461|1|0.1539|
|0|1|0|1|0.5425|0.9971|0.0385| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.3974| |1.6052|0.8327|0|-0.8327|
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
#### Derivatives
|Slope hidden layer|||d_hidden_layer|||Slope output|d_output|
|:---:|---|---|:---:|---|---|:---:|:---:|
|0.0928|0.1355|0.1205|0.00003167|0.001937|0.0001082|0.1373|0.0226| 
|0.0531|0.0688|0.0795|0.00001604|0.0008711|0.0006319|0.1302|0.0200| 
|0.1081|0.1736|0.0861|-0.0001893|-0.012742|-0.003969|0.1393|-0.1160|

7. Finally update weights and biases at both the hidden as well as the output layers (assuming a fixed learning rate of 1 for this exercise for simplicity). The learning rate is a hyperparamater that is tuned based on requirements for each model. 
```
  wout = wout + matrix_dot_product(hidden_layer_activations.Transpose, d_output) * learning_rate
  wh = wh + matrix_dot_product(X.Tranpose, d_hidden_layer) * learning_rate
  bout = bout + sum(d_output, axis = 0) * learning_rate
  bh = bh + sum(d_hidden_layer, axis = 0) * learning_rate
  ```
#### Values
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.8317|0.5788|0.8152|0.7847|0.0694|0.9605|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|0.0151| 0.7409|1.6269|0.8357|1|0.1643|
|1|0|1|1|0.5168|0.2988|0.7560| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.6331| |1.7040|0.8461|1|0.1539|
|0|1|0|1|0.5425|0.9971|0.0385| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.3974| |1.6052|0.8327|0|-0.8327|
| ||||0.6603|0.8766|0.5348| ||| ||| ||| | | | | | |
#### Derivatives
|Slope hidden layer|||d_hidden_layer|||Slope output|d_output|
|:---:|---|---|:---:|---|---|:---:|:---:|
|0.0928|0.1355|0.1205|0.00003167|0.001937|0.0001082|0.1373|0.0226| 
|0.0531|0.0688|0.0795|0.00001604|0.0008711|0.0006319|0.1302|0.0200| 
|0.1081|0.1736|0.0861|-0.0001893|-0.012742|-0.003969|0.1393|-0.1160|
### New values
|X||||wh|||bh|||hidden_layer_input|||hidden_layer_activations|||wout|bout|output_layer_input|output|y|E|
|:-----:|-----|-----|-----|:---:|---|----|:---:|---|---|:---:|---|---|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
|1|0|1|0|0.83175|0.58161|0.81594|0.78456|0.05947|0.95727|2.1589|1.6453|1.8142|0.8965|0.8383|0.8599|-0.04746| 0.6675|1.6269|0.8357|1|0.1643|
|1|0|1|1|0.51661|0.28606|0.75203| |||2.8192|2.5219|2.3490|0.9437|0.9257|0.9129|0.58049| |1.7040|0.8461|1|0.1539|
|0|1|0|1|0.54255|0.99991|0.04131| |||1.9618|1.2448|2.2513|0.8767|0.7764|0.9048|0.33013| |1.6052|0.8327|0|-0.8327|
| ||||0.66013|0.87472|0.53146| ||| ||| ||| | | | | | |

8. All the above steps are repeated iteratively until the loss(E) reduces and reaches a minimum value.
 
Note: Part A of the assignment can be found at this [link](https://github.com/shreyasrajesh/MLBLR-EIP) 