import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from math import sqrt
from pathlib import Path
import json

class MODEL:
    
    def __init__(self, m, m_test, neurons, activations, epochs, batch_size, learning_rate, regularization_param, optimization_param, optimizer = "momentum"):
        self.X_train, self.Y_train, self.m = None, None, np.minimum(m, 60000)
        self.X_test, self.Y_test, self.m_test = None, None, np.minimum(m_test, 10000)
        self.neurons = neurons
        self.activations = activations
        self.forward_activations = []
        self.reverse_activations = []
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_count = None
        self.alpha_ = learning_rate
        self.lambda_ = regularization_param
        self.beta = optimization_param
        self.x_train_flatten, self.x_test_flatten= None, None
        self.weights = {}
        self.biases = {}
        self.train_acc, self.test_acc = None, None
        self.optimizer = optimizer

    # loading dataset
    def loadData(self):
        (self.X_train, self.Y_train), (self.X_test, self.Y_test) = mnist.load_data()

    # shuffling the data and slicing given amount
    def shuffleAndSliceData(self, x, y, size):
        shuffle = np.random.permutation(len(x))
        x_shuffled = x[shuffle]
        y_shuffled = y[shuffle]
        x_new = x_shuffled[0:size, :]
        y_new = y_shuffled[0:size]
    
        return x_new, y_new
    
    # Converting the each images' pixels as a one vector => we get X matrix each column is one image.
    def vectorize(self, x):
        vectorized_x = x.reshape(x.shape[0], -1).T
        return vectorized_x
    
    # Normalizing input values because of the ensure numerical stability, this function provides normal distribution
    def zScoreNormalize(self, data):
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        normalized_data = (data - mean_val) / std_val
        
        return normalized_data
    
    # creating new Y matrix with one hot encoding because our target values categorical
    def oneHotY(self, Y):
        one_hot_Y = np.zeros((10, Y.size)) 
        one_hot_Y[Y, np.arange(Y.size)] = 1     # (10, Y.size)

        return one_hot_Y
    
    # preprocess to get ready the data for training
    def preProcess(self):
        # loading data
        self.loadData()
        
        # shuffling and slicing the data
        self.X_train, self.Y_train = self.shuffleAndSliceData(self.X_train, self.Y_train, self.m)
        self.X_test, self.Y_test = self.shuffleAndSliceData(self.X_test, self.Y_test, self.m_test)
        
        # converting to each image as a one vector
        self.x_train_flatten, self.x_test_flatten = self.vectorize(self.X_train), self.vectorize(self.X_test)
        
        # normalizing the image pixel values to prevent bigger numbers, big numbers cause computational problems
        self.x_train_flatten, self.x_test_flatten = self.zScoreNormalize(self.x_train_flatten), self.zScoreNormalize(self.x_test_flatten)
        
        # creating one hot "y" data because of our model is categorical
        self.y_one_hot_train, self.y_one_hot_test = self.oneHotY(self.Y_train), self.oneHotY(self.Y_test)
        
        # adding pixel amount for initial layer
        self.neurons.insert(0, 784)
        
        # finding how many batch requires for the given batch size
        batch_count = lambda x, y: x // y + 1 if x % y != 0 else x // y
        self.batch_count = batch_count(self.m, self.batch_size)

    # split to X and Y given batch size, mini batch generator
    def miniBatch(self):
        batch_count = self.m // self.batch_size

        for slice in range(batch_count):
            X_sliced = self.x_train_flatten[:, slice * self.batch_size : (slice+1) * self.batch_size]
            Y_sliced = self.Y_train[slice * self.batch_size : (slice+1) * self.batch_size]
            y_one_hot_train_sliced = self.y_one_hot_train[:, slice * self.batch_size : (slice+1) * self.batch_size]
            yield X_sliced, Y_sliced, y_one_hot_train_sliced

        if self.m % self.batch_size != 0 :
            # last slice which not fit with given batch size    
            X_sliced = self.x_train_flatten[:, batch_count * self.batch_size:]
            Y_sliced = self.Y_train[batch_count * self.batch_size:]
            y_one_hot_train_sliced = self.y_one_hot_train[:, batch_count * self.batch_size:]
            yield X_sliced, Y_sliced, y_one_hot_train_sliced

    # HE initialization
    def heInit(self, index):
        self.weights[f"W{index+1}"] = np.random.randint(10, size=(self.neurons[index+1], self.neurons[index])) * 0.1 * np.sqrt(2./self.neurons[index])
        self.biases[f"b{index+1}"] = np.random.randint(10, size=(self.neurons[index+1], 1)) * 0.1
    
    # XAVIER initialization
    def xavierInit(self, index):
        lower, upper = -(1.0 / sqrt(self.neurons[index])), (1.0 / sqrt(self.neurons[index]))
        self.weights[f"W{index+1}"] = lower + np.random.randint(10, size=(self.neurons[index+1], self.neurons[index])) * 0.1 * (upper - lower)
        self.biases[f"b{index+1}"] = np.random.randint(10, size=(self.neurons[index+1], 1)) * 0.1

    # initializing parameters
    def initParams(self):
        ind = 0

        for i in self.activations:
            # HE initialization for Relu Activation Funtions
            if i == 'relu':
                self.heInit(ind)
            # XAVIER initialization for softmax activation function
            elif i == 'softmax':
                self.xavierInit(ind)

            ind += 1

        # optimization parameters
        opt_cache = {}
        for i in range(len(self.activations)):
            opt_cache[f"opt_dW{i+1}"] = np.zeros_like(self.weights[f"W{i+1}"])
            opt_cache[f"opt_db{i+1}"] = np.zeros_like(self.biases[f"b{i+1}"])

        return opt_cache
    
    # Rectifier linear unit function for activation
    def relu(self, Z):
        return np.maximum(0, Z)

    # Derivative of relu
    def reluDerivative(self, Z):
        return Z > 0

    # Softmax is last layer activation function for our digits case
    def softmax(self, Z):
        e_z = np.exp(Z - np.max(Z))
        return e_z / e_z.sum(axis=0)
    
    # setting up chosen activation functions
    def functionChooser(self):
        
        for i in self.activations:
            if i == 'relu':
                self.forward_activations.append(self.relu)
                self.reverse_activations.append(self.reluDerivative)
            elif i == 'softmax':
                self.forward_activations.append(self.softmax)
                self.reverse_activations.append(0)

        if self.optimizer == "momentum":
            self.optimizer = self.gradientMomentum
        else:
            pass
        
        
    # forward propogation
    def forwardProp(self, X):
        cache = {}
        cache["A0"] = X
        
        for i in range(len(self.forward_activations)):
            cache[f"Z{i+1}"] = np.dot(self.weights[f"W{i+1}"], cache[f"A{i}"]) + self.biases[f"b{i+1}"]
            func = self.forward_activations[i]
            cache[f"A{i+1}"] = func(cache[f"Z{i+1}"])
            
        return cache
    
    # backward propagation
    def backwardProp(self, forward_cache, Y):
        m = Y.shape[0]
        cache = {}
        cache[f"dLdZ_{3}"] = forward_cache[f"A{3}"] - Y
        cache[f"dLdW_{3}"] = np.dot(cache[f"dLdZ_{3}"], forward_cache[f"A{2}"].T) / m
        cache[f"dLdb_{3}"] = np.sum(cache[f"dLdZ_{3}"]) / m

        for i in range(len(self.reverse_activations)-1, 0, -1):
            func = self.reverse_activations[i-1]
            cache[f"dLdZ_{i}"] = np.dot(self.weights[f"W{i+1}"].T, cache[f"dLdZ_{i+1}"]) * func(forward_cache[f"Z{i}"])
            cache[f"dLdW_{i}"] = np.dot(cache[f"dLdZ_{i}"], forward_cache[f"A{i-1}"].T) / m
            cache[f"dLdb_{i}"] = np.sum(cache[f"dLdZ_{i}"]) / m

        return cache
    
    # gradient momentum
    def gradientMomentum(self, opt_cache, backward_cache):
    
        for i in range(len(self.activations)):
            opt_cache[f"opt_dW{i+1}"] = self.beta * opt_cache[f"opt_dW{i+1}"] + (1 - self.beta) * backward_cache[f"dLdW_{i+1}"]
            opt_cache[f"opt_db{i+1}"] = self.beta * opt_cache[f"opt_db{i+1}"] + (1 - self.beta) * backward_cache[f"dLdb_{i+1}"]
             
        return opt_cache
      
    # parameter updating with L2 Regularization ( (lambda_ / m) * W[i] ) and gradient momentum parameters V_dw, W_db
    def updateParams(self, opt_cache, m):
        
        for i in range(len(self.activations)):
            self.weights[f"W{i+1}"] -= self.alpha_ * (opt_cache[f"opt_dW{i+1}"] + (self.lambda_ / m) * self.weights[f"W{i+1}"])
            self.biases[f"b{i+1}"] -= self.alpha_ * opt_cache[f"opt_db{i+1}"]

    # Loss computing
    def computeLoss(self, A3, Y):
        m = Y.shape[1]
        log_probs = -np.log(A3[Y == 1])
        loss = np.sum(log_probs) / m

        return loss
    
    # model last layer predictions
    def predict(self, A3):
        return np.argmax(A3, 0)

    # accuracy of model last layer predictions
    def accuracy(self, predictions, Y):  
        return np.sum(predictions == Y) / Y.size * 100
    
    # gradient descent algorithm with mini-batching
    def gradientDescent(self):
        opt_cache = self.initParams()
        losses = []

        for i in range(self.epochs+1):
            # to compute avg accuracy and avg loss summing all batches' loss and acc values then dividing batch amount
            loss = 0
            train_acc = 0

            # slicing X and Y matrices to given batch size then applying forward prop, backward pro and update params to all batches
            for X_sliced, Y_sliced, y_one_hot_train_sliced in self.miniBatch():
                forward_cache = self.forwardProp(X_sliced)
                backward_cache = self.backwardProp(forward_cache, y_one_hot_train_sliced)
                opt_cache = self.optimizer(opt_cache, backward_cache)
                self.updateParams(opt_cache, Y_sliced.size)
                loss += self.computeLoss(forward_cache["A3"], y_one_hot_train_sliced)
                predictions = self.predict(forward_cache["A3"])
                train_acc += self.accuracy(predictions, Y_sliced)

            loss /= self.batch_count
            train_acc /= self.batch_count
            losses.append(loss)

            if i % 20 == 0:
                print(f"---EPOCH {i}---")
                print(f"Accuracy= %{train_acc:.5f}")
                print(f"Loss= {loss:.5f}\n")
                
        self.train_acc = train_acc

        return losses
    
    def trainModel(self):
        self.preProcess()
        self.functionChooser()
        losses = self.gradientDescent()
        self.testSetAccuracy()
        
        return losses
    
    # visualizing loss values
    def plotLoss(self, loss):
        plt.plot(loss)
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.title("Loss vs Epoch Graph")
        plt.show()
        
    # predicting value of given image(s)
    def singlePredict(self, single_X):
        cache = self.forwardProp(single_X)
        prediction = self.predict(cache["A3"])

        return prediction
    
    #testing image prediction is True or False
    def testPredict(self, index):
        current_image = self.x_train_flatten[:, index, None]
        prediction = self.singlePredict(current_image)
        true_label = self.Y_train[index]

        return prediction == true_label, prediction, true_label
    
    # finding all wrong labeled images or true predictions with amount 25
    def getPredictions(self, type, set_type = "train"):
        predicts_sample = []

        if set_type == "train":
            size = self.m
        
        elif set_type == "test":
            size = self.m_test
        
        if type.lower() == "wrong":

            for i in range(size):
                isSame, prediction, true_label = self.testPredict(i)
                if(isSame == False):
                    predicts_sample.append([i, prediction, true_label])

        elif type.lower() == "true":

            while len(predicts_sample) < 25:
                i = np.random.randint(self.m)
                isSame, prediction, true_label = self.testPredict(i)
                if(isSame == True):
                    predicts_sample.append([i, prediction, true_label])

        self.plotPredictions(type, predicts_sample, set_type)
        return predicts_sample
    
    
    # plotting prediction samples and their true labels
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
                axes[i, j].imshow(self.X_train[ind, :])
                axes[i, j].set_title(f"i: {ind}, P: {p}, T: {t}")

        plt.tight_layout()
        plt.show()
        
    # Printing index, prediction and true label of given sample
    def testPredictionSample(self, sample):

        for i in range(len(sample)):
            _, p, t = self.testPredict(sample[i])
            print(f"Index: {sample[i]}, Prediction: {p}, True Label: {t}")
            
    def testSetAccuracy(self):
        self.test_acc = self.accuracy(self.singlePredict(self.x_test_flatten), self.Y_test)
        
        print(f"\nTest Set Accuracy: %{self.test_acc:.5f}")
            
    # Finding probability distribution of one image
    def probability(self, index):
        cache = self.forwardProp(self.x_train_flatten[:, index, None])

        return cache["A3"]

    # Finding all probabilities of given index list
    def getProbabilities(self, index_list):
        probs = []

        for i in index_list:
            probs.append(self.probability(i))

        return probs
    
    # Plotting wrong labeled images and their probabilities 
    def plotProbabilities(self, index_list):
        probs = self.getProbabilities(index_list)
        size = 5
        fig, axes = plt.subplots(size, 2)
        fig.set_figheight(10)
        fig.set_figwidth(10)
        plt.gray()
        plt.suptitle("Model Prediction Probabilities of Wrong Labeled Images in Test Set")

        for i in range(size):
            true_label = self.Y_train[index_list[i]]
            axes[i, 0].imshow(self.X_train[index_list[i], :])
            axes[i, 0].set_title(f"i: {index_list[i]}, True Label: {true_label}")
            axes[i, 1].set_xlabel("Digits")
            axes[i, 1].set_ylabel("Probability %")
            axes[i, 1].bar(range(10), probs[i].squeeze() * 100, width = 0.4, color = "purple")      # squeeze benefits the fix array dimensions
            plt.sca(axes[i, 1])
            plt.xticks(range(10))
            plt.yticks(range(0, 101, 20))

        plt.tight_layout()
        plt.show()
        
    # saving model informations on a txt and json file
    def saveModelData(self, type_='3-nn-v3'):
        model_results = Path(__file__).with_name('trained_models.txt')
        model_infos = Path(__file__).with_name('trained_models_properties.json')

        model_ex = {"id": None, "weights": {"w1": self.W1.tolist(), "w2": self.W2.tolist(), "w3": self.W3.tolist()}, "biases": {"b1": self.b1.tolist(), "b2": self.b2.tolist(), "b3": self.b3.tolist()}}

        with model_results.open('a+') as f:
            f.seek(0)
            lines = f.readlines()
            model_id = len(lines) - 1
            model_ex["id"] = model_id
            f.write(f"{model_id:^9}|| {type_:^11}|| {str(self.epochs)+' epochs':^16}|| {self.m:^21}|| {self.m_test:^17}|| {self.alpha_:^14}|| {self.lambda_:^7}|| {self.train_acc:^15.5f}|| {self.test_acc:^14.5f}|| {'batch+'+self.optimizer:^15}|| {self.batch_size:^11}||\n")
            f.close()

        with open(model_infos, "r+") as f:
            data = json.load(f)
            data.append(model_ex)
            f.seek(0)
            json.dump(data, f, indent=4)
            f.close()
    
    
m_train, m_test = 60000, 10000
neurons = [20, 15, 10]
activations = ['relu', 'relu', 'softmax']
epochs = 100
batch_size = 128
learning_rate = 0.01
lambda_ = 0.1
beta = 0.9

model = MODEL(m_train, m_test, neurons, activations, epochs, batch_size, learning_rate, lambda_, beta)
# train the model
loss = model.trainModel()

# plotting loss values
model.plotLoss(loss)

# random indices to visualize model guess and their true values
test_indices = []
for i in range(10):
    rand = np.random.randint(m_train)
    test_indices.append(rand)

model.testPredictionSample(test_indices)

# plotting some of true and wrong predictions with maxixum amount of 25
model.getPredictions("true")
wrong_label_train_set = model.getPredictions("wrong") # you can get all wrong labeled image indexes
print(f"train wrong amount: {len(wrong_label_train_set)}")

# test set wrong labeled images
wrong_label_test_set = model.getPredictions("wrong", set_type = "test")
print(f"test wrong amount: {len(wrong_label_test_set)}")

# bar chart visualization of probability distribution of wrong labeled images in test set (sample amount = 5)
test_indices = []
for i in range(5):
    rand = np.random.randint(len(wrong_label_test_set))
    test_indices.append(wrong_label_test_set[rand][0])

model.plotProbabilities(test_indices)

# saving model findings
isSave = input("Do you want to save this model? (Y/N) ")
if isSave.lower() == 'y':
    model.saveModelData()
