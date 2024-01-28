# NEURAL NETWORK with DIGITS DATASET
&nbsp;

## DATASET
- Images are **(8,8)** pixels
- Total image amount is **1797**


&nbsp;
## 3 LAYER NEURAL NETWORK v1
- 1st layer -> 20 neurons, **RELU** activation function
- 2nd layer -> 15 neurons, **RELU** activation function
- 3rd layer -> 10 neurons, **SOFTMAX** activation function
- All layers **fully connected**
- Iteration amount: **10000**
- Learning Rate: **0.01**
- There is no optimizer, initializer just vanilla neural network with **Z Score Normalization**
- Lack of the optimizer provides **unstability** on model performance
- Lack of the initializer provides **not learning** model
- Accuracy around **%75 - %90** in general

&nbsp;
&nbsp;
**Last Iteration of One Model**

![ScreenShot](./doc/3_layer_nn_v1/3_layer_nn_v1_accuracy.png)

&nbsp;
&nbsp;
**Testing Some Sample**

![ScreenShot](./doc/3_layer_nn_v1/3_layer_nn_v1_testing_some_sample.png)

&nbsp;
&nbsp;
**A True Labeled Sample from Model Predictions**

![ScreeShot](./doc/3_layer_nn_v1/3_layer_nn_v1_true_sample.png)

&nbsp;
&nbsp;
**A Wrong Labeled Sample from Model Predictions**

![ScreeShot](./doc/3_layer_nn_v1/3_layer_nn_v1_wrong_sample.png)




