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
	""" A Gaussian Process Dynamic Model. Two GPs are used - one for the observation mapping and one for the dynamic mapping."""
	def __init__(self,Y,dim):
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
		
	def filter(self,nparticles=100):
		"""Inference of the state/latent variable using Sequential Monte carlo.
		
		Currently uses a simple sample-importance-resample procedure. Could be update to include mcmc jumping or similar"""
		X = np.zeros((self.T,nparticles,self.Xdim))
		
		#sample for x0 TODO: variable priors on X0
		X[0,:,:] = np.random.randn(nparticles,self.Xdim)
		weights = np.ones(nparticles)
		for t in range(1,self.T):
			#sample
			mu,var = self.dynamic_GP.predict(X[t-1,:,:])
			X[t,:,:] = np.random.randn(nparticles,self.Xdim)*np.sqrt(var) + mu # TODO check dimensions here?
			
			#importance
			ypred,predvar = self.observation_GP.predict(X[t,:,:])
			weights = 0.5*np.sum(np.square(ypred-self.observation_GP.Y[t])/predvar)
			weights /= weights.sum()
			
			#resample
			# TODO: use some criteria as to whether resampling is necessary
			index = np.random.multinomial(nparticles,weights)
			X[t,:,:] = X[t,:,:].repeat(index,axis=0)
			
			
			
			
		
		