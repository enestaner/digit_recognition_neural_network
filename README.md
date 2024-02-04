# NEURAL NETWORK with DIGITS DATASET
&nbsp;

## DATASET
- Images are **(28, 28)** pixels
- MNIST dataset has **60000** training, **10000** test images


&nbsp;
## 3 LAYER NEURAL NETWORK v1
- 1st layer -> 20 neurons, **RELU** activation function
- 2nd layer -> 15 neurons, **RELU** activation function
- 3rd layer -> 10 neurons, **SOFTMAX** activation function
- All layers **fully connected**
- Training Image Amount: **10000**
- Iteration amount: **10000**
- Learning Rate: **0.01**
- There is no optimizer and initializer, just vanilla neural network with **Z Score Normalization**
- Lack of the optimizer provides **unstability** on model performance
- Lack of the initializer provides **not learning** model
- Accuracy around **%75 - %85** in general

&nbsp;
**Last Iteration of One Model**

![ScreenShot](./img/3_layer_nn_v1/3_layer_nn_v1_accuracy.png)

&nbsp;
**Testing Some Sample with Random Indices**

![ScreenShot](./img/3_layer_nn_v1/3_layer_nn_v1_testing_some_sample.png)

&nbsp;
**A True Labeled Sample from Model Predictions**

![ScreenShot](./img/3_layer_nn_v1/3_layer_nn_v1_true_sample.png)

&nbsp;
**A Wrong Labeled Sample from Model Predictions**

![ScreenShot](./img/3_layer_nn_v1/3_layer_nn_v1_wrong_sample.png)

&nbsp;
## 3 LAYER NEURAL NETWORK v2
- 1st layer -> 20 neurons, **RELU** activation function
- 2nd layer -> 15 neurons, **RELU** activation function
- 3rd layer -> 10 neurons, **SOFTMAX** activation function
- All layers **fully connected**
- Training Image Amount: **15000**, more than these causes slow computation
- Test Image Amount: **2500**
- Iteration amount: **3000**
- Learning Rate = **0.01**
- There is **He Initializer** for RELU, **Xavier Initializer** for SOFTMAX activation functions
- There is **L2 Regularization** for prevent overfitting
- Regularization parameter **Lambda = 0.05**
- Still there is no optimizer so, **unstable** gradient descending shows up
- Training set accuracy around **%90+** in general
- Test set accuracy between **%89 - %92** in general
#### *Differences between v1 and v2*
- Because of the initializers the accuracy is **so rapidly** increasing
- Because of the L2 regularization **overfitting is decreasing** and test set **accuracy is increasing**
- No need for lots of iterations anymore, **2000 - 3000 iterations** enough for this model to learn

&nbsp;
**Last Iteration of One Model**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_train_accuracy.png)

&nbsp;
**Test Set Accuracy**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_test_accuracy.png)

&nbsp;
**A True Labeled Sample from Train Set**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_train_true_sample.png)

&nbsp;
**A Wrong Labeled Sample from Train Set**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_train_wrong_sample.png)

&nbsp;
**A Wrong Labeled Sample from Test Set**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_test_wrong_sample.png)

&nbsp;
**Loss Graph**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_loss.png)

&nbsp;
**Model Prediction Probabilities of Wrong Labeled Images in Test Set**

![ScreenShot](./img/3_layer_nn_v2/3_layer_nn_v2_probs.png)