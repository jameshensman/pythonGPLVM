# -*- coding: utf-8 -*-
# all kernels expect numpy arrays of data.i
# Arrays must have two dimensions: the first for the Number of data, the second for the dimension of the data.  

import numpy as np

class RBF:
	"""Radial Basis Funcion (or 'Squared Exponential') kernel, with the same scale in all directions...
	k(x_i,x_j) = \alpha \exp \{ -\gamma ||x_1-x_2||^2 \}
	"""
	def __init__(self,alpha,gamma):
		self.alpha = np.exp(alpha)
		self.gamma = np.exp(gamma)
		self.nparams = 2
		
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.alpha,self.gamma = np.exp(new_params).copy().flatten()#try to unpack np array safely.  
		
	def get_params(self):
		#return np.array([self.alpha, self.gamma])
		return np.log(np.array([self.alpha, self.gamma]))
		
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		#use broadcasting to avoid for loops. 
		#should be uber fast
		diff = x1.reshape(N1,1,D1)-x2.reshape(1,N2,D2)
		diff = self.alpha*np.exp(-np.sum(np.square(diff),-1)*self.gamma)
		return diff
		
	def gradients(self,x1):
		"""Calculate the gradient of the matrix K wrt the (log of the) free parameters"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		diff = np.sum(np.square(diff),-1)
		#dalpha = np.exp(-diff*self.gamma)
		dalpha = self.alpha*np.exp(-diff*self.gamma)
		#dgamma = -self.alpha*diff*np.exp(-diff*self.gamma)
		dgamma = -self.alpha*self.gamma*diff*np.exp(-diff*self.gamma)
		return (dalpha, dgamma)
		
	def gradients_wrt_data(self,x1,indexn=None,indexd=None):
		"""compute the derivative matrix of the kernel wrt the _data_. Crazy
		This returns a list of matices: each matrix is NxN, and there are N*D of them!"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		diff = np.sum(np.square(diff),-1)
		expdiff = np.exp(-self.gamma*diff)
		
		if (indexn==None) and(indexd==None):#calculate all gradients
			rets = []
			for n in range(N1):
				for d in range(D1):
					K = np.zeros((N1,N1))
					K[n,:] = -2*self.alpha*self.gamma*(x1[n,d]-x1[:,d])*expdiff[n,:]
					K[:,n] = K[n,:]
					rets.append(K.copy())
			return rets
		else:
			K = np.zeros((N1,N1))
			K[indexn,:] = -2*self.alpha*self.gamma*(x1[indexn,indexd]-x1[:,indexd])*expdiff[indexn,:]
			K[:,indexn] = K[indexn,:]
			return K

class RBF_full:
	def __init__(self,alpha,gammas):
		self.gammas = np.exp(gammas.flatten())
		self.dim = gammas.size
		self.alpha = np.exp(alpha)
		self.nparams = self.dim+1
		
	def set_params(self,params):
		assert params.size==self.nparams)
		self.alpha = np.exp(params.flatten()[0])
		self.gammas = np.exp(params.flatten()[1:])
		
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		assert D1==self.dim, "That data does not match the dimensionality of this kernel"
		diff = x1.reshape(N1,1,D1)-x2.reshape(1,N2,D2)
		diff = self.alpha*np.exp(-np.sum(np.square(diff)*self.gammas,-1))
		return diff
		
	def gradients(self,x1):
		"""Calculate the gradient of the matrix K wrt the (log of the) free parameters"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		sqdiff = np.sum(np.square(diff)*self.gamma,-1)
		expdiff = np.exp(-sqdiff)
		grads = [-g*np.square(diff[:,:,i])*self.alpha*expdiff for i,g in enumerate(self.gammas)]
		dgamma.insert(0, self.alpha*np.expdiff))
		return dgamma
	
	def gradients_wrt_data(self,x1):
		"""compute the derivative matrix of the kernel wrt the _data_. Crazy
		This returns a list of matices: each matrix is NxN, and there are N*D of them!"""
		N1,D1 = x1.shape
		diff = x1.reshape(N1,1,D1)-x1.reshape(1,N1,D1)
		sqdiff = np.sum(np.square(diff)*self.gammas,-1)
		expdiff = np.exp(-sqdiff)
		
		if (indexn==None) and(indexd==None):#calculate all gradients
			rets = []
			for n in range(N1):
				for d in range(D1):
					K = np.zeros((N1,N1))
					K[n,:] = -2*self.alpha*self.gammas[d]*(x1[n,d]-x1[:,d])*expdiff[n,:]
					K[:,n] = K[n,:]
					rets.append(K.copy())
			return rets
		else:
			K = np.zeros((N1,N1))
			K[indexn,:] = -2*self.alpha*self.gammas[indexd]*(x1[indexn,indexd]-x1[:,indexd])*expdiff[indexn,:]
			K[:,indexn] = K[indexn,:]
			return K.copy()
		
	
			
		
		
class linear:
	"""effectively the inner product, I think"""
	def __init__(self,alpha,bias):
		self.alpha = alpha
		self.bias = bias
		self.nparams = 2
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.alpha,self.bias = new_params.flatten()#try to unpack np array safely.  
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		prod = x1.reshape(N1,1,D1)*x2.reshape(1,N2,D2)
		prod = self.alpha*np.sum(prod,-1) + self.bias
		#diff = self.alpha*np.sqrt(np.square(np.sum(diff,-1)))
		return prod
	def gradients(self,x1):
		"""Calculate the gradient of the kernel matrix wrt the parameters"""
		dalpha = (self(x1,x1)-self.bias)/self.alpha
		dbias = np.ones((x1,shape[0],x1.shape[0]))
		return dalpha, dbias

class combined:
	""" a combined kernel - linear in X and RBF in Y.  
	treats first Dimensiona linearly, RBf on the remainder. 
	TODO: specify which dimensions should be linear and which should be RBF"""
	def __init__(self,alpha_x,alpha_y,gamma,bias):
		self.linear_kernel = linear(alpha_x, bias)
		self.RBF_kernel = RBF(alpha_y, gamma)
		self.nparams = 4
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.linear_kernel.set_params(new_params[:2])
		self.RBF_kernel.set_params(new_params[2:])
		
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		return self.linear_kernel(x1[:,0:1],x2[:,0:1])*self.RBF_kernel(x1[:,1:],x2[:,1:])
		
class polynomial:
	def __init__(self,alpha,order):
		"""Order of the polynomila is considered fixed...TODO: make the order optimisable..."""
		self.alpha = alpha
		self.order = order
		self.nparams = 1
	def set_params(self,new_params):
		assert new_params.size == self.nparams
		self.alpha, = new_params.flatten()
	def __call__(self,x1,x2):
		N1,D1 = x1.shape
		N2,D2 = x2.shape
		assert D1==D2, "Vectors must be of matching dimension"
		prod = x1.reshape(N1,1,D1)*x2.reshape(1,N2,D2)
		prod = self.alpha*np.power(np.sum(prod,-1) + 1, self.order)
		return prod
		