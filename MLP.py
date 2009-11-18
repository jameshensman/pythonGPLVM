# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
import numpy as np
from scipy import optimize

class MLP:
	def __init__(self,structure,output='linear',alpha=1):
		assert len(structure) == 3
		self.structure = structure
		self.alpha = alpha # regulariser/prior on weights
		self.nweights = (structure[0]+1)*structure[1]+(structure[1]+1)*structure[2]
		
		self.initialise()
		
		if output=='linear':
			self.output_fn = lambda x:x
			self.error_fn = lambda y,t : 0.5*np.sum(np.square(y-t))
		
	def initialise(self):
		"""Initialise the weights of the network by sampling from the (Gaussian) prior"""
		nin,nhid,nout = self.structure
		s = 1./np.sqrt(self.alpha)
		self.w1 = np.random.randn(nin,nhid)*s
		self.b1 = np.random.randn(1,nhid)*s
		self.w2 = np.random.randn(nhid,nout)*s
		self.b2 = np.random.randn(1,nout)*s
	
	def forward(self,x):
		"""calculate the outputs of the network given a set of inputs x"""
		n,d = x.shape
		assert d == self.structure[0], "Input dimension does not match this network"
		self.activations = np.tanh(np.dot(x,self.w1) + np.ones((n,1))*self.b1)
		self.out = np.dot(self.activations,self.w2) + np.ones((n,1))*self.b2
		return self.output_fn(self.out)
	
	def prior(self):
		"""evaluate the current set of weights under the prior = return the log value"""
		return -0.5*self.alpha*np.sum(np.square(self.pack()))
		
	def prior_grad(self):
		"""evaluate the gradient of the prior at the current set of weights"""
		return -self.alpha*self.pack()
		
	def gradient(self,weights,x,t):
		"""used for training"""
		self.unpack(weights)
		y = self.forward(x)
		delta_out = y-t #gradient of the error wrt output, y
		return self.backpropagate(x,delta_out) - self.prior_grad()
	
	def backpropagate(self,x,delta_out):
		"""'backpropagate' the gradeint of the error wrt to the output of the network to the gradient of the error wrt the weights
		Essentially doing de/dw = de/dy*dy/dw""" 
		
		#Evaluate second-layer gradients.
		gw2 = np.dot(self.activations.T,delta_out)
		gb2 = np.sum(delta_out, 0)
		
		# Backpropagation to hidden layer.
		delta_hid = np.dot(delta_out,self.w2.T)
		delta_hid *= (1.0 - self.activations**2)
		
		# Finally, evaluate the first-layer gradients.
		gw1 = np.dot(x.T,delta_hid)
		gb1 = np.sum(delta_hid, 0)

		return np.hstack([e.flatten() for e in [gw1,gb1,gw2,gb2]])
	
	def error(self,weights,x,t):
		"The thing to be optimised in training"""
		self.unpack(weights)
		y = self.forward(x)
		return self.error_fn(y,t) - self.prior()
		
	def train(self,x,t):
		w = optimize.fmin_cg(self.error,self.pack(),fprime=self.gradient,args=(x,t))
		self.unpack(w)
		
	def unpack(self,weights):
		"""take a np array and assign it to self.w1 etc."""
		weights = weights.flatten()
		self.w1 = weights[:self.w1.size].reshape(self.w1.shape)
		self.b1 = weights[self.w1.size:self.w1.size+self.b1.size].reshape(self.b1.shape)
		self.w2 = weights[self.w1.size+self.b1.size:self.w1.size+self.b1.size+self.w2.size].reshape(self.w2.shape)
		self.b2 = weights[-self.b2.size:].reshape(self.b2.shape)
	
	def pack(self):
		""" 'Pack up' the weights and biases into a vector"""
		return np.hstack([e.flatten() for e in [self.w1,self.b1,self.w2,self.b2]])
		
		
		

if __name__=='__main__':
	import pylab
	
	N = 50
	x = np.random.randn(N,1)
	y = np.sin(2*x) + np.random.randn(N,1)*0.1
	
	xx = np.linspace(-2,2,100).reshape(100,1)
	myMLP = MLP((1,5,1),alpha=0.1)
	
	pylab.plot(x,y,'ro')
	pylab.plot(xx,myMLP.forward(xx))
	
	w = optimize.fmin(myMLP.error,myMLP.pack(),args=(x,y))
	myMLP.unpack(w)
	pylab.plot(xx,myMLP.forward(xx))
	
	myMLP.initialise()
	w = optimize.fmin_cg(myMLP.error,myMLP.pack(),fprime=myMLP.gradient,args=(x,y))
	myMLP.unpack(w)
	pylab.plot(xx,myMLP.forward(xx))
	
	
	pylab.show()