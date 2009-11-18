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
		self.N,self.Ydim = Y.shape
		
		"""Use PCA to initalise the problem. Uses EM version in this case..."""
		myPCA_EM = PCA_EM(Y,dim)
		myPCA_EM.learn(300)
		X = np.array(myPCA_EM.m_Z)
		
		self.observation_GP = GP.GP(X,Y)
		self.dynamic_GP = GP.GP(X[:-1],X[1:])
		
		