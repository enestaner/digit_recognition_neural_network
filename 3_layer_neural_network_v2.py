import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from math import sqrt
from pathlib import Path
import json, codecs

#Normalizing input values because of the ensure numerical stability, this function provides normal distribution
def zScoreNormalize(data):
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

#creating new Y matrix with one hot encoding because our target values categorical
def oneHotY(Y):
    one_hot_Y = np.zeros((np.max(Y) + 1, Y.size)) 
    one_hot_Y[Y, np.arange(Y.size)] = 1     #(10, 15000)

    return one_hot_Y

def initParams():
    #HE initialization for Relu Activation Funtions
    W1 = np.random.randint(10, size=(20, 784)) * 0.1 * np.sqrt(2./784)
    b1 = np.random.randint(10, size=(20, 1)) * 0.1
    W2 = np.random.randint(10, size=(15, 20)) * 0.1 * np.sqrt(2./20)
    b2 = np.random.randint(10, size=(15, 1)) * 0.1
    #Xavier initialization for softmax activation function
    lower, upper = -(1.0 / sqrt(15)), (1.0 / sqrt(15))
    W3 = lower + np.random.randint(10, size=(10, 15)) * 0.1 * (upper - lower)
    b3 = np.random.randint(10, size=(10, 1)) * 0.1

    return W1, b1, W2, b2, W3, b3

#Rectifier linear unit function for activation
def relu(Z):
    return np.maximum(0, Z)

#Derivative of relu
def reluDerivative(Z):
    return Z > 0

#Softmax is last layer activation function for our digits case
def softmax(Z):
    e_z = np.exp(Z - np.max(Z))
    return e_z / e_z.sum(axis=0)

#forward propogation
def forwardProp(W1, b1, W2, b2, W3, b3, X):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    
    return Z1, A1, Z2, A2, Z3, A3

#backward propagation
def backwardProp(Z1, Z2, A1, A2, A3, W2, W3, X, Y):
    one_hot_Y = oneHotY(Y)
    m_train = X.shape[0]
    dLdZ_3 = A3 - one_hot_Y
    dLdW_3 = np.dot(dLdZ_3, A2.T) / m_train
    dLdb_3 = np.sum(dLdZ_3) / m_train
    dLdZ_2 = np.dot(W3.T, dLdZ_3) * reluDerivative(Z2)
    dLdW_2 = np.dot(dLdZ_2, A1.T) / m_train
    dLdb_2 = np.sum(dLdZ_2) / m_train
    dLdZ_1 = np.dot(W2.T, dLdZ_2) * reluDerivative(Z1)
    dLdW_1 = np.dot(dLdZ_1, X.T) / m_train
    dLdb_1 = np.sum(dLdZ_1) / m_train
    
    return dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3

#parameter updating with L2 Regularization ( (lambda_ / m_train) * W[i] )
def updateParams(W1, W2, W3, b1, b2, b3, dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3, learning_rate, lambda_, m_train):
    W1 -= learning_rate * (dLdW_1 + (lambda_ / m_train) * W1)  
    b1 -= learning_rate * dLdb_1
    W2 -= learning_rate * (dLdW_2 + (lambda_ / m_train) * W2)
    b2 -= learning_rate * dLdb_2
    W3 -= learning_rate * (dLdW_3 + (lambda_ / m_train) * W3)
    b3 -= learning_rate * dLdb_3
       
    return W1, b1, W2, b2, W3, b3

#Loss computing
def computeLoss(A3, Y):
    m_train = Y.size
    Y = oneHotY(Y)
    log_probs = -np.log(A3[Y == 1])
    loss = np.sum(log_probs) / m_train

    return loss

#model last layer predictions
def predict(A3):
    return np.argmax(A3, 0)

#accuracy of model last layer predictions
def accuracy(predictions, Y):  
    return np.sum(predictions == Y) / Y.size * 100

#gradient descent algorithm and model itself
def gradientDescent(X, Y, iter, learning_rate, lambda_):
    W1, b1, W2, b2, W3, b3 = initParams()
    losses = []
    
    for i in range(iter+1):
        Z1, A1, Z2, A2, Z3, A3 = forwardProp(W1, b1, W2, b2, W3, b3, X)
        dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3 = backwardProp(Z1, Z2, A1, A2, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = updateParams(W1, W2, W3, b1, b2, b3, dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3, learning_rate, lambda_, Y.size)
        loss = computeLoss(A3, Y)
        losses.append(loss)
        
        if i % 100 == 0:
            predictions = predict(A3)
            acc = accuracy(predictions, Y)
            print(f"--Iteration {i}--")
            print(f"Prediction -> {predictions}")
            print(f"Real Values -> {Y}")
            print(f"Accuracy= %{acc:.5f}")
            print(f"Loss= {loss:.5f}\n")
    
    return W1, b1, W2, b2, W3, b3, losses, acc

#visualizing loss values
def plotLoss(loss):
    
    plt.plot(loss)
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.title("Loss vs Iteration Graph")
    plt.show()


class MODEL:
    def __init__(self, X, Y, W1, b1, W2, b2, W3, b3):
        self.X = X
        self.Y = Y
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.W3 = W3
        self.b3 = b3

    #predicting value of given image(s)
    def singlePredict(self, single_X):
        _, _, _, _, _, A3 = forwardProp(self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, single_X)
        prediction = predict(A3)
        
        return prediction
    
    #testing image prediction is True or False
    def testPredict(self, index):
        current_image = self.X[:, index, None]
        prediction = self.singlePredict(current_image)
        true_label = self.Y[index]
    
        return prediction == true_label, prediction, true_label
    
    #finding all wrong labeled images or true predictions with amount 25
    def predictionSamples(self, type, set_type = "train"):
        predicts_sample = []
        
        if type.lower() == "wrong":
            
            for i in range(self.Y.size):
                isSame, prediction, true_label = self.testPredict(i)
                if(isSame == False):
                    predicts_sample.append([i, prediction, true_label])
                    
        elif type.lower() == "true":
            
            while len(predicts_sample) < 25:
                i = np.random.randint(self.Y.size)
                isSame, prediction, true_label = self.testPredict(i)
                if(isSame == True):
                    predicts_sample.append([i, prediction, true_label])
                    
        self.plotPredictions(type, predicts_sample, set_type)
        return predicts_sample
    
    #plotting prediction samples and their true labels
    def plotPredictions(self, type, sample, set_type):
        size = 5
        fig, axes = plt.subplots(size, size)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        plt.gray()
        plt.suptitle(f"A Sample of {type.capitalize()} Predictions from {set_type.capitalize()} Set\nIndex:i || Model Prediction: P || True Label: T")
        
        for i in range(size):
            for j in range(size):
                try:
                    ind, p, t = sample[i * size + j]
                except IndexError:
                    i = size+1
                    break
                axes[i, j].imshow(self.X[:, ind, None].reshape((28, 28)))
                axes[i, j].set_title(f"i: {ind}, P: {p}, T: {t}")
        
        plt.tight_layout()
        plt.show()
        
    #Printing index, prediction and true label of given sample
    def testPredictionSample(self, sample):
    
        for i in range(len(sample)):
            _, p, t = self.testPredict(sample[i])
            print(f"Index: {sample[i]}, Prediction: {p}, True Label: {t}")
    
    #Finding probability distribution of one image
    def probability(self, index):
        _, _, _, _, _, prob = forwardProp(self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.X[:, index, None])
        
        return prob
    
    #Finding all probabilities of given index list
    def getProbabilities(self, index_list):
        probs = []
        
        for i in index_list:
            probs.append(self.probability(i))
            
        return probs
    
    #Plotting wrong labeled images and their probabilities 
    def plotProbabilities(self, index_list):
        probs = self.getProbabilities(index_list)
        size = 5
        fig, axes = plt.subplots(size, 2)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.gray()
        plt.suptitle("Model Prediction Probabilities of Wrong Labeled Images in Test Set")
        
        for i in range(size):
            true_label = self.Y[index_list[i]]
            axes[i, 0].imshow(self.X[:, index_list[i], None].reshape((28, 28)))
            axes[i, 0].set_title(f"i: {index_list[i]}, True Label: {true_label}")
            axes[i, 1].set_xlabel("Digits")
            axes[i, 1].set_ylabel("Probability %")
            axes[i, 1].bar(range(10), probs[i].squeeze() * 100, width = 0.4, color = "purple")
            plt.sca(axes[i, 1])
            plt.xticks(range(10))
            plt.yticks(range(0, 101, 20))
            
        plt.tight_layout()
        plt.show()
    
    #saving model informations on a txt file
    def saveModelData(self, iter_count, train_count, test_count, train_acc, test_acc, learning_rate, lambda_, type_="3-nn-v2"):
        model_results = Path(__file__).with_name('trained_models.txt')
        model_infos = Path(__file__).with_name('trained_models_properties.json')
        
        model_ex = {"id": None, "weights": {"w1": self.W1.tolist(), "w2": self.W2.tolist(), "w3": self.W3.tolist()}, "biases": {"b1": self.b1.tolist(), "b2": self.b2.tolist(), "b3": self.b3.tolist()}}

        with model_results.open('a+') as f:
            f.seek(0)
            lines = f.readlines()
            model_id = len(lines) - 1
            model_ex["id"] = model_id
            f.write(f"{model_id:^9}|| {type_:^11}|| {iter_count:^16}|| {train_count:^21}|| {test_count:^17}|| {learning_rate:^14}|| {lambda_:^7}|| {train_acc:^15.5f}|| {test_acc:^13.5f}\n")
            f.close()
        
        with open(model_infos, "r+") as f:
            data = json.load(f)
            data.append(model_ex)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.close()
                
#getting MNIST Digits dataset from keras library
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train[0:15000, :]   #(15000, 28, 28)
y_train = y_train[0:15000]      #(15000, )
m_train = y_train.size          # 15000
x_test = x_test[0:2500, :]      #(2500, 28, 28)
y_test = y_test[0:2500]         #(2500, )
m_test = y_test.size            # 2500

# Converting the each images' pixels as a one vector => we get X matrix each column is one image.
x_train_flat = x_train.reshape(x_train.shape[0], -1).T  #(784, 15000)
x_test_flat = x_test.reshape(x_test.shape[0], -1).T    #(784, 2500)

# Normalizing input data
x_train_flat = zScoreNormalize(x_train_flat)
x_test_flat = zScoreNormalize(x_test_flat)

#training model
learning_rate = 0.01
lambda_ = 0.05
iters = 3000
W1, b1, W2, b2, W3, b3, losses, train_accuracy = gradientDescent(x_train_flat, y_train, iters, learning_rate, lambda_)

#plotting loss values
plotLoss(losses)

#creating 'model' object with weights and biases which model has found. We can easily predict or visualize our findings. 
model = MODEL(x_train_flat, y_train, W1, b1, W2, b2, W3, b3)

#random indices to visualize model guess and their true values
test_indices = []
for i in range(10):
    rand = np.random.randint(m_train)
    test_indices.append(rand)
    
model.testPredictionSample(test_indices)

#plotting some of true and wrong predictions with maxixum amount of 25
model.predictionSamples("true")
wrong_label_train_set = model.predictionSamples("wrong") #you can get all wrong labeled image indexes

#test set accuracy
model.X, model.Y = x_test_flat, y_test
test_accuracy = accuracy(model.singlePredict(model.X), model.Y)
print(f"\nTest Set Accuracy: %{test_accuracy:.5f}")

#test set wrong labeled images
wrong_label_test_set = model.predictionSamples("wrong", set_type = "test")

#bar chart visualization of probability distribution of wrong labeled images in test set (sample amount = 5)
test_indices = []
for i in range(5):
    rand = np.random.randint(len(wrong_label_test_set))
    test_indices.append(wrong_label_test_set[rand][0])
    
model.plotProbabilities(test_indices)

#saving model findings
isSave = input("Do you want to save this model? (Y/N) ")
if isSave.lower() == 'y':
    model.saveModelData(iters, m_train, m_test, train_accuracy, test_accuracy, learning_rate, lambda_)
