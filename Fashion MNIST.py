# -*- coding: utf-8 -*-


import tensorflow as tf
from tensorflow import keras
tf.__version__
keras.__version__

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test_full, y_test_full) = fashion_mnist.load_data()

print(X_train_full.shape)
X_valid, X_train = X_train_full[:5000] / 255.0 , X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000] , y_train_full[5000:]

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
"Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
class_names[y_train[0]]

#Creating the model to learn the data 
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))                    #Converts all the images in a 1 dimensional array(i.e. stacks them vertically)
model.add(keras.layers.Dense(300, activation = "relu"))                   #Creates a layer of 300 neuron units with activation function being ReLu(REctified Linear Unit)
model.add(keras.layers.Dense(100, activation = "relu"))                   #Creates a layer of 100 neuron units with activation function being ReLu(REctified Linear Unit)
model.add(keras.layers.Dense(10, activation = "softmax"))                 #Creates final output layer consisting of 10 neuron units each unit corresponding to the 10 classes having activation function softmax

model.summary()

#compiling the model
#The following line compiles the model and runs it where the 
#Loss function is given by "sparse categorical crossentropy" function
#The function used to optimize or approach global minima of Loss function is "Stochastic Gradient Descent(SGD)"
#The metric on which the perfomance of the model is measured is "Accuracy"
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "sgd", metrics = ["accuracy"])

history = model.fit(X_train, y_train, epochs = 30, validation_data = (X_valid, y_valid))





