# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
#from numpy import matlib as ml
import numpy as np
from scipy import linalg

class PCA_EM_matrix:
	def __init__(self,data,target_dim):
		"""Maximum likelihood PCA by the EM algorithm"""
		self.X = ml.matrix(data)
		self.N,self.d = self.X.shape
		self.q = target_dim
	def learn(self,niters):
		self.mu = self.X.mean(0).reshape(self.d,1)#ML solution for mu
		self.X2 = self.X - self.mu.T
		self.xxTsum = ml.sum([x*x.T for x in self.X2])#precalculate for speed
		#initialise paramters:
		self.W = ml.randn(self.d,self.q)
		self.sigma2 = 1.2
		for i in range(niters):
			#print self.sigma2
			self.E_step()
			self.M_step()

	def E_step(self):
		M = self.W.T*self.W + ml.eye(self.q)*self.sigma2
		M_inv = ml.linalg.inv(M)
		self.m_Z = (M_inv*self.W.T*self.X2.T).T
		self.S_z = M_inv*self.sigma2
	def M_step(self):
		zzT = self.m_Z.T*self.m_Z + self.N*self.S_z
		self.W = self.X2.T*self.m_Z*ml.linalg.inv(zzT)
		WTW = self.W.T*self.W
		self.sigma2 = self.xxTsum - 2*ml.multiply(self.m_Z*self.W.T,self.X2).sum() + ml.trace(zzT*WTW)
		#self.sigma2 = self.xxTsum - 2*ml.trace(self.m_Z*self.W.T*self.X2.T) + ml.trace(zzT*WTW)
		#self.sigma2 = self.xxTsum + ml.sum([- 2*z*self.W.T*x.T + ml.trace((z.T*z + self.S_z)*WTW) for z,x in zip(self.m_Z, self.X2)])
		self.sigma2 /= self.N*self.d
		
class PCA_EM:
	def __init__(self,data,target_dim):
		"""Maximum likelihood PCA by the EM algorithm"""
		self.X = np.array(data)
		self.N,self.d = self.X.shape
		self.q = target_dim
	def learn(self,niters):
		self.mu = self.X.mean(0).reshape(self.d,1)#ML solution for mu
		self.X2 = self.X - self.mu.T
		self.xxTsum = np.sum([np.dot(x,x.T) for x in self.X2])#precalculate for speed
		#initialise paramters:
		self.W = np.random.randn(self.d,self.q)
		self.sigma2 = 1.2
		for i in range(niters):
			#print self.sigma2
			self.E_step()
			self.M_step()

	def E_step(self):
		M = np.dot(self.W.T,self.W) + np.eye(self.q)*self.sigma2
		#M_inv = np.linalg.inv(M)
		#self.m_Z = np.dot(M_inv,np.dot(self.W.T,self.X2.T)).T
		#self.S_z = M_inv*self.sigma2
		M_chol = linalg.cholesky(M)
		M_inv = linalg.cho_solve((M_chol,1),np.eye(self.q))
		self.m_Z = linalg.cho_solve((M_chol,1),np.dot(self.W.T,self.X2.T)).T
		self.S_z = M_inv*self.sigma2
		
	def M_step(self):
		zzT = np.dot(self.m_Z.T,self.m_Z) + self.N*self.S_z
		#self.W = np.dot(np.dot(self.X2.T,self.m_Z),np.linalg.inv(zzT))
		zzT_chol = linalg.cholesky(zzT)
		self.W = linalg.cho_solve((zzT_chol,0),np.dot(self.m_Z.T,self.X2)).T
		WTW = np.dot(self.W.T,self.W)
		self.sigma2 = self.xxTsum - 2*np.sum(np.dot(self.m_Z,self.W.T)*self.X2) + np.trace(np.dot(zzT,WTW))
		self.sigma2 /= self.N*self.d
		
class PCA_EM_missing:
	def __init__(self,data,target_dim):
		"""Maximum likelihood PCA by the EM algorithm, allows for missing data.  uses a masked array to 'hide' the elements of X that are NaN"""
		self.X = np.array(data)
		self.Xmasked = np.ma.masked_array(data,np.ma.make_mask(np.isnan(self.X)))
		self.N,self.d = self.X.shape
		self.q = target_dim
		
	def learn(self,niters):
		self.mu = np.asarray(self.Xmasked.mean(0)).reshape(self.d,1)#ML solution for mu
		#use a masked array to decide which data we're learning from.
		self.X2masked = self.Xmasked - self.mu.T
		#make an unmasked copy of X (to be filled with the reconstructed values)
		self.X2unmasked = np.array(self.Xmasked).copy()
		self.imask,self.jmask = np.nonzero(np.isnan(self.X2unmasked))
		self.X2unmasked[self.imask,self.jmask] = 0
		#initialise paramters:
		self.W = np.random.randn(self.d,self.q)
		self.sigma2 = 1.2
		#pre-allocate self.m_Z and self.S_Z 
		self.m_Z = np.zeros((self.X2masked.shape[0],self.q))
		self.S_Z = np.zeros((self.X2masked.shape[0],self.q,self.q))
		for i in range(niters):
			print i,self.sigma2
			self.E_step()
			self.M_step()

	def E_step(self):
		""" This should handle missing data, but needs testing (TODO)"""
		Ms = np.zeros((self.X2masked.shape[0],self.q,self.q)) #M is going to be different for (potentially) every data point
		for m,x,i in zip(Ms,self.X2masked,np.arange(self.X2masked.shape[0])):
			index = np.nonzero(x.mask-1)[0]#select non masked parts
			W = self.W.take(index,0)# get relevant bits of W
			x2 = np.array(x).take(index) # get relevant bits of x
			m[:,:] = np.dot(W.T,W) + np.eye(self.q)*self.sigma2
			mchol = linalg.cholesky(m)
			minv = linalg.cho_solve((mchol,1),np.eye(self.q))
			self.m_Z[i,:] = linalg.cho_solve((mchol,1),np.dot(W.T,x2.reshape(index.size,1))).T
			self.S_Z[i,:,:] = minv*self.sigma2
			
			#calculate reconstructed X values
			Xreconstruct = np.dot(self.m_Z,self.W.T)
			self.X2unmasked[self.imask,self.jmask] = Xreconstruct[self.imask,self.jmask]
			self.xxTsum = np.sum(np.square(self.X2unmasked))# can;t be pre-calculate in the missing data version :(
		
		#M = np.dot(self.W.T,self.W) + np.eye(self.q)*self.sigma2
		#M_chol = linalg.cholesky(M)
		#M_inv = linalg.cho_solve((M_chol,1),np.eye(self.q))
		#self.m_Z = linalg.cho_solve((M_chol,1),np.dot(self.W.T,self.X2.T)).T
		#self.S_z = M_inv*self.sigma2
		
	def M_step(self):
		""" This should handle missing data - needs testing (TODO)"""
		zzT = np.dot(self.m_Z.T,self.m_Z) + np.sum(self.S_Z,0)
		#self.W = np.dot(np.dot(self.X2.T,self.m_Z),np.linalg.inv(zzT))
		zzT_chol = linalg.cholesky(zzT)
		self.W = linalg.cho_solve((zzT_chol,0),np.dot(self.m_Z.T,self.X2unmasked)).T
		WTW = np.dot(self.W.T,self.W)
		self.sigma2 = self.xxTsum - 2*np.sum(np.dot(self.m_Z,self.W.T)*self.X2unmasked) + np.trace(np.dot(zzT,WTW))
		self.sigma2 /= self.N*self.d

if __name__=='__main__':
	q=2#latent dimensions
	d=15# observed dimensions
	N=500
	Nmissing = 490
	truesigma = .02
	niters = 300
	phases = np.random.rand(1,q)*2*np.pi
	latents = np.sin(np.linspace(0,12,N).reshape(N,1)*np.ones((1,q))-phases)
	trueW = np.random.randn(d,q)
	observed = np.dot(latents,trueW.T) + np.random.randn(N,d)*truesigma
	a = PCA_EM(observed,q)
	a.learn(niters)
	
	#a missing data problem
	observed2 = observed.copy()
	missingi = np.argsort(np.random.rand(N))[:Nmissing]
	missingj = np.random.randint(0,d-q,Nmissing)#last q columns will be complete
	observed2[missingi,missingj] = np.NaN
	
	b = PCA_EM_missing(observed2,q)
	b.learn(niters)
	
	
	from hinton import hinton
	import pylab
	colours = np.arange(N)# to colour the dots with
	hinton(linalg.qr(trueW.T)[1].T)
	pylab.title('true transformation')
	pylab.figure()
	hinton(linalg.qr(a.W.T)[1].T)
	pylab.title('reconstructed transformation')
	pylab.figure()
	hinton(linalg.qr(b.W.T)[1].T)
	pylab.title('reconstructed transformation (missing data)')
	pylab.figure()
	pylab.subplot(3,1,1)
	pylab.plot(latents)
	pylab.title('true latents')
	pylab.subplot(3,1,2)
	pylab.plot(a.m_Z)
	pylab.title('reconstructed latents')
	pylab.subplot(3,1,3)
	pylab.plot(b.m_Z)
	pylab.title('reconstructed latents (missing data)')
	pylab.figure()
	pylab.subplot(2,1,1)
	pylab.plot(observed)
	pylab.title('Observed values')
	pylab.subplot(2,1,2)
	pylab.plot(observed2,linewidth=2,marker='.')
	pylab.plot(b.X2unmasked + b.mu.T)
	#pylab.subplot(2,1,1)
	#pylab.plot(observed,marker='.')
	#pylab.subplot(2,1,2)
	#pylab.plot(observed2,'.')

	pylab.show()

