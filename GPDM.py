# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
#
# Gaussian Process Dynamic Model
import numpy as np
import pylab
from PCA_EM import PCA_EM
import kernels
import GP
from scipy import optimize


class GPDM:
	""" A Gaussian Process Dynamic Model. Two GPs are used - one for the observation mapping and one for the dynamic mapping.
	A particle filter is used for inference of the latent variables.  
	
	In the EM-like procedure, the expected values of the latent variables are used for the GP data.  This is obviously not strictly correct, but stochasic GP regression (i.e. with samples for the X data) is beyond me right now. """
	def __init__(self,Y,dim,nparticles=100):
		self.Xdim = dim
		self.T,self.Ydim = Y.shape
		
		"""Use PCA to initalise the problem. Uses EM version in this case..."""
		myPCA_EM = PCA_EM(Y,dim)
		myPCA_EM.learn(300)
		X = np.array(myPCA_EM.m_Z)
		
		self.observation_GP = GP.GP(X,Y)
		
		#create a linear kernel for the dynamics
		k = kernels.linear(-1,-1)
		self.dynamic_GP = GP.GP(X[:-1],X[1:],k)
		
		#initialise the samples from the state /latent variables
		self.particles = np.zeros((self.T,nparticles,self.Xdim))
		
		#sample for x0 TODO: variable priors on X0
		self.particles[0,:,:] = np.random.randn(nparticles,self.Xdim)
		
	def filter(self,X,Y):
		"""Inference of the state/latent variable using Sequential Monte Carlo.
		Currently uses a simple sample-importance-resample procedure. Could be update to include mcmc jumping or similar
		Takes arguments X and Y, so that is can be used for inference bith in learning (self.X,self.Y) and in use.  
		
		X is a TxNxD array, T=time, N= number of particles, D= dimension of the state
		Y is a TxQ aray: T=time,Q is dimension of observed space"""
		
		weights = np.ones(X.shape[1])
		for t in range(1,self.T):
			#sample
			mu,var = self.dynamic_GP.predict(X[t-1,:,:])
			X[t,:,:] = np.random.randn(nparticles,self.Xdim)*np.sqrt(var) + mu # TODO check dimensions here?
			
			#importance
			ypred,predvar = self.observation_GP.predict(X[t,:,:])
			weights = 0.5*np.sum(np.square(ypred-Y[t])/predvar)
			weights /= weights.sum()
			
			#resample
			# TODO: use some criteria as to whether resampling is necessary
			index = np.random.multinomial(X.shape[1],weights)
			X[t,:,:] = X[t,:,:].repeat(index,axis=0)
		return X
		
	def set_GP_params(self,params):
		"""set the parameters of th two GPs to the passed values"""
		assert params.size == self.obsertation_GP.kernel.nparams_self.dynamic_GP.kernel.nparams +2, "Bad number fo parameters for setting"
		self.observation_GP.set_params(params[:self.observation_GP.kernel.nparams+1])
		self.dynamic_GP.set_params(params[self.observation_GP.kernel.nparams+1:])
		
	def get_GP__params(self):
		return np.hstack((self.observation_GP.get_params(),self.dynamic_GP.get_params()))
		
	def learn(self,iters):
		for i in range(iters):
			self.filter(self.particles,self.observation_GP.Y)
			# TODO reverse filter.
			EX = self.particles.mean(1) # the expected value of the latent variables.
			self.observation_GP.X = EX
			self.dynamic_GP.X,self.dynamic_GP.Y = EX[:-1],EX[1:]
			self.dynamic_GP.find_kernel_parameters()
			self.observation_GP.find_kernel_parameters()
			
			
		
		
		
		
			
			
			
			
		
		