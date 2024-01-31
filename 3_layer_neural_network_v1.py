import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

#Normalizing input values because of the ensure numerical stability, this function provides normal distribution
def zScoreNormalize(data):
    mean_val = np.mean(data, axis=0)
    std_val = np.std(data, axis=0)
    normalized_data = (data - mean_val) / std_val
    return normalized_data

#creating new Y matrix with one hot encoding because our target values categorical
def oneHotY(Y):
    one_hot_Y = np.zeros((np.max(Y) + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1     #(10,1797)

    return one_hot_Y

#[0, 1) initialization of weights and biases nothing special, this will change in other versions 
def initParams():
    W1 = np.random.randint(10, size=(20, 784)) * 0.1
    b1 = np.random.randint(10, size=(20, 1)) * 0.1
    W2 = np.random.randint(10, size=(15, 20)) * 0.1
    b2 = np.random.randint(10, size=(15, 1)) * 0.1
    W3 = np.random.randint(10, size=(10, 15)) * 0.1
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

#parameter updating
def updateParams(W1, W2, W3, b1, b2, b3, dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3, learning_rate):
    W1 -= learning_rate * dLdW_1
    b1 -= learning_rate * dLdb_1
    W2 -= learning_rate * dLdW_2
    b2 -= learning_rate * dLdb_2
    W3 -= learning_rate * dLdW_3
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
def gradientDescent(X, Y, iter, learning_rate):
    W1, b1, W2, b2, W3, b3 = initParams()
    losses = []
    
    for i in range(iter+1):
        Z1, A1, Z2, A2, Z3, A3 = forwardProp(W1, b1, W2, b2, W3, b3, X)
        dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3 = backwardProp(Z1, Z2, A1, A2, A3, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = updateParams(W1, W2, W3, b1, b2, b3, dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3, learning_rate)
        loss = computeLoss(A3, Y)
        
        if i % 100 == 0:
            losses.append(loss)
        
        if i % 200 == 0:
            predictions = predict(A3)
            acc = accuracy(predictions, Y)
            print(f"--Iteration {i}--")
            print(f"Prediction -> {predictions}")
            print(f"Real Values -> {Y}")
            print(f"Accuracy= %{acc:.5f}")
            print(f"Loss= {loss:.5f}\n")
    
    return W1, b1, W2, b2, W3, b3, losses

#visualization of loss values
def plotLoss(loss):
    
    plt.plot(loss[1:])
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.title("Loss vs Iteration Graph")
    plt.show()

def randomTestIndices(amount, max_index):
    test_indices = []
    
    for i in range(amount):
        test_indices.append(np.random.randint(max_index))
        
    return test_indices

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
        
    def singlePredict(self, single_X):
        _, _, _, _, _, A3 = forwardProp(self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, single_X)
        prediction = predict(A3)
        
        return prediction
    
    def testPredict(self, index):
        current_image = self.X[:, index, None]
        prediction = self.singlePredict(current_image)
        true_label = self.Y[index]
    
        return prediction == true_label, prediction, true_label
    
    def predictionSamples(self, type):
        predicts_sample = []
        
        if type.lower() == "wrong":
            
            for i in range(self.Y.size):
                isSame, prediction, true_label = self.testPredict(i)
                if(isSame == False):
                    predicts_sample.append([i, prediction, true_label])
                    
        elif type.lower() == "true":
            
            while len(predicts_sample) < 25:
                i = np.random.randint(10000)
                isSame, prediction, true_label = self.testPredict(i)
                if(isSame == True):
                    predicts_sample.append([i, prediction, true_label])
                    
        self.plotPredictions(type, predicts_sample)
        return predicts_sample
    
    def plotPredictions(self, type, sample):
        size = 5
        fig, axes = plt.subplots(size, size)
        fig.set_figheight(8)
        fig.set_figwidth(8)
        plt.gray()
        plt.suptitle(f"A Sample of {type.capitalize()} Predictions\nIndex:i || Model Prediction: P || True Label: T")
        
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
        
    def testPredictionSample(self, sample):
    
        for i in range(len(sample)):
            _, p, t = self.testPredict(sample[i])
            print(f"Index: {sample[i]}, Prediction: {p}, True Label: {t}")     
        
#getting MNIST Digit data from keras library
(x_train, y_train), (_,_) = mnist.load_data()

x_train = x_train[0:10000, :]   #(10000, 28, 28)
y_train = y_train[0:10000]      #(10000,)
m_train = y_train.size

# Converting the each images' pixels as a one vector => we get X matrix each column is one image.
x_flat = x_train.reshape(x_train.shape[0], -1).T    #(784, 10000)

# Normalizing data  
x_flat = zScoreNormalize(x_flat)

#training model
W1, b1, W2, b2, W3, b3, losses = gradientDescent(x_flat, y_train, 10000, 0.01)

#plotting loss values
plotLoss(losses)

#creating 'model' object with weights and biases which model has found. We can easily predict or visualize our findings. 
model = MODEL(x_flat, y_train, W1, b1, W2, b2, W3, b3)

#random indices to visualize model's guess and their true values
test_indices = randomTestIndices(10, m_train)
model.testPredictionSample(test_indices)

#plotting some of true and wrong predictions with maxixum amount of 25
model.predictionSamples("true")
wrong_label = model.predictionSamples("wrong") #you can get all wrong labeled image indexes
