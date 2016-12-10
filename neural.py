import numpy
import scipy.special
import scipy.optimize
import sklearn.metrics

#Compatibility with old sklearn
import sklearn
from distutils.version import LooseVersion
if LooseVersion(sklearn.__version__) < LooseVersion("0.18"):
	from sklearn.cross_validation import train_test_split
else:
	from sklearn.model_selection import train_test_split
import os.path

import config
from util import debug

sigmoid = scipy.special.expit

def sigmoid_gradient(z):
	y = sigmoid(z)
	return y*(1-y)

class NeuralClassfier:
	def __init__(self,input_layer,hidden_layer,outer_layer,reg,random_seed=None,w_range=config.w_range):
		self.input_layer = input_layer
		self.hidden_layer = hidden_layer
		self.outer_layer = outer_layer
		self.reg = reg
		self.w_range = w_range
		if(random_seed):
			numpy.random.seed(random_seed)
		self.weights=numpy.random.random(self.hidden_layer*(self.input_layer+1)+self.outer_layer*(self.hidden_layer+1))*2*self.w_range - self.w_range

	def layer_matrix(self,weights=None):
		if weights is None:
			weights = self.weights
		Theta1=weights[:(self.input_layer+1)*self.hidden_layer].reshape((self.input_layer+1,self.hidden_layer)).T
		Theta2=weights[(self.input_layer+1)*self.hidden_layer:].reshape((self.hidden_layer+1,self.outer_layer)).T
		return (Theta1,Theta2)

	def predict(self,X):
		if(len(X.shape) == 1):
			X=numpy.reshape(X,(1,-1))
		Theta1,Theta2 = self.layer_matrix(self.weights)

		set_size = X.shape[0]
		P = numpy.concatenate(( numpy.ones((set_size,1)), X ), axis=1)
		P = sigmoid(numpy.dot(P,Theta1.T))

		P = numpy.concatenate((numpy.ones((set_size,1)),P), axis=1)
		P = sigmoid(numpy.dot(P,Theta2.T))
		return P

	def get_cost_function(self,X,Y):
		def ret(weights = self.weights):
			Theta1,Theta2 = self.layer_matrix(weights)
			
			set_size = X.shape[0]
			a1 = numpy.concatenate(( numpy.ones((set_size,1)), X ), axis=1)

			z2 = a1.dot(Theta1.T)
			a2 = numpy.concatenate((numpy.ones((set_size,1)),sigmoid(z2)), axis=1)

			z3 = a2.dot(Theta2.T)
			a3 = sigmoid(z3)
			h = a3
			
			J = (-1/set_size)*numpy.sum(Y*numpy.log(h)+(1-Y)*numpy.log(1-h))+self.reg*1/(2*set_size)*(numpy.sum(numpy.square(Theta1[:,1:]))+numpy.sum(numpy.square(Theta2[:,1:])));
			return J
		return ret

	def get_gradient(self,X,Y):
		def ret(weights = self.weights):
			Theta1,Theta2 = self.layer_matrix(weights)

			m = X.shape[0]
			a1 = numpy.concatenate(( numpy.ones((m,1)), X ), axis=1)

			z2 = a1.dot(Theta1.T)
			a2 = numpy.concatenate((numpy.ones((m,1)),sigmoid(z2)), axis=1)

			z3 = a2.dot(Theta2.T)
			a3 = sigmoid(z3)
			
			delta3=(a3-Y);

			delta2=delta3.dot(Theta2)[:,1:]*sigmoid_gradient(z2)
			
			Theta1Zero = numpy.concatenate((numpy.zeros((Theta1.shape[0],1)),Theta1[:,1:]),axis=1)
			Theta2Zero = numpy.concatenate((numpy.zeros((Theta2.shape[0],1)),Theta2[:,1:]),axis=1)

			Theta1_grad=1/m*delta2.T.dot(a1)+self.reg*1/m*Theta1Zero
			Theta2_grad=1/m*delta3.T.dot(a2)+self.reg*1/m*Theta2Zero
			return numpy.concatenate((Theta1_grad.T.flatten(),Theta2_grad.T.flatten()),axis=0)
		return ret

	def learn(self,X,Y,callback=None,maxiter=config.maxiter,method=config.method):
		J=self.get_cost_function(X,Y)
		Gr=self.get_gradient(X,Y)
		#self.weights = scipy.optimize.fmin_cg(J,self.weights,fprime=Gr,callback=callback,maxiter=maxiter)
		res = scipy.optimize.minimize(J,self.weights,method=method,jac=Gr,callback=callback,options={ "maxiter" : maxiter })
		self.weights = res.x
		return self.weights

	def dataset_accuracy(self,X,Y):
		Yp=self.predict(X)
		yp = one_vs_all_to_class_number(Yp)
		y = one_vs_all_to_class_number(Y)
		accuracy = sklearn.metrics.accuracy_score(y,yp)
		return accuracy

	def get_log(self,X_tr,Y_tr,X_val,Y_val):
		counter = -1
		def log(W):
			nonlocal counter
			self.weights = W
			counter += 1
			if(counter % 20 != 0): return
			debug("Iteration: {}".format(counter))
			debug("Cost: {}".format(self.get_cost_function(X_tr,Y_tr)(W)))
			debug("Training accuracy: {}".format(self.dataset_accuracy(X_tr,Y_tr)))
			debug("Validation accuracy: {}".format(self.dataset_accuracy(X_val,Y_val)))
			debug("")
		return log

def class_number_to_one_vs_all(y,outer_layer):
	train_size = y.shape[0]
	Y = numpy.zeros((train_size,outer_layer))
	for i in range(train_size):
		Y[i][y[i]]=1
	return Y

def one_vs_all_to_class_number(Y):
	#numpy.where(Y==1)[1]
	return numpy.argmax(Y,axis=1)

def train_network(X,y):
	'''Train new network'''
	input_layer = X.shape[1]
	outer_layer = config.character_number

	neural_classifier = NeuralClassfier(input_layer,config.hidden_layer,outer_layer,config.reg,random_seed=config.seed)
	
	X_tr,X_val,y_tr,y_val = train_test_split(X,y,random_state=config.seed)

	Y_tr = class_number_to_one_vs_all(y_tr,outer_layer)
	Y_val = class_number_to_one_vs_all(y_val,outer_layer)
	
	logger = neural_classifier.get_log(X_tr,Y_tr,X_val,Y_val)

	weights = neural_classifier.learn(X_tr,Y_tr,callback=logger)
	logger(weights)
	
	return neural_classifier
