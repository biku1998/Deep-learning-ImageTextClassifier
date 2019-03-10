# Deep-learning-ImageTextClassifier
given an image , classify whether if contains Text or not.

# How Does it works : 

## As you can see there are 2 folders that contains images for test and train.

## The model that is used is Sigmoid Neuron with perceptron model.

## 2 Loss functions are there to use : 
	1. mean squared error loss
	2. cross entropy loss

## Flow : 
	The images are read from the folders converted into numpy array and then split into 
	test and train modules.Then the train module is passed into the fit function of the model.
	There is also accuracy printed w.r.t to both loss function.

# You can tune the HyperParameter i.e learning_rate and epochs to see how the model performs.
