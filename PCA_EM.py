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
		self.imask,self.jmask = np.nonzero(np.isnan(self.X))#positions that are missing.
		self.indices = [np.nonzero(np.isnan(x)-1)[0] for x in self.X] #positions that are not missing...
		self.N,self.d = self.X.shape
		self.q = target_dim
		
	def learn(self,niters):
		self.Xreconstruct = self.X.copy()
		self.Xreconstruct[self.imask,self.jmask] = 0
		self.mu = np.sum(self.Xreconstruct,0)/(self.X.shape[0]-np.sum(np.isnan(self.X),0))
		
		self.X2 = self.X.copy()-self.mu
		self.X2reconstruct = self.X.copy() - self.mu
		#initialise paramters:
		self.W = np.random.randn(self.d,self.q)
		self.sigma2 = 1.2
		#pre-allocate self.m_Z and self.S_Z 
		self.m_Z = np.zeros((self.X2.shape[0],self.q))
		self.S_Z = np.zeros((self.X2.shape[0],self.q,self.q))
		for i in range(niters):
			print i,self.sigma2
			self.E_step()
			self.M_step()
		self.Xreconstruct = self.X2reconstruct + self.mu

	def E_step(self):
		""" This should handle missing data, but needs testing (TODO)"""
		Ms = np.zeros((self.X.shape[0],self.q,self.q)) #M is going to be different for (potentially) every data point
		for m,x,i,mz,sz in zip(Ms,self.X2,self.indices,self.m_Z,self.S_Z):
			W = self.W.take(i,0)# get relevant bits of W
			x2 = np.array(x).take(i) # get relevant bits of x
			m[:,:] = np.dot(W.T,W) + np.eye(self.q)*self.sigma2
			mchol = linalg.cholesky(m)
			minv = linalg.cho_solve((mchol,1),np.eye(self.q))
			mz[:] = linalg.cho_solve((mchol,1),np.dot(W.T,x2.reshape(i.size,1))).T
			sz[:,:] = minv*self.sigma2
			
		#calculate reconstructed X values
		self.X2reconstruct[self.imask,self.jmask] = np.dot(self.m_Z,self.W.T)[self.imask,self.jmask]
		self.xxTsum = np.sum(np.square(self.X2reconstruct))# can;t be pre-calculate in the missing data version :(
		
	def M_step(self):
		""" This should handle missing data - needs testing (TODO)"""
		zzT = np.dot(self.m_Z.T,self.m_Z) + np.sum(self.S_Z,0)
		#self.W = np.dot(np.dot(self.X2.T,self.m_Z),np.linalg.inv(zzT))
		zzT_chol = linalg.cholesky(zzT)
		self.W = linalg.cho_solve((zzT_chol,0),np.dot(self.m_Z.T,self.X2reconstruct)).T
		WTW = np.dot(self.W.T,self.W)
		self.sigma2 = self.xxTsum - 2*np.sum(np.dot(self.m_Z,self.W.T)*self.X2reconstruct) + np.trace(np.dot(zzT,WTW))
		self.sigma2 /= self.N*self.d

if __name__=='__main__':
	q=5#latent dimensions
	d=15# observed dimensions
	N=500
	missing_pc = 100 # percentage of the data points to be 'missing'
	truesigma = .002
	niters = 300
	phases = np.random.rand(1,q)*2*np.pi
	frequencies = np.random.randn(1,q)*2
	latents = np.sin(np.linspace(0,12,N).reshape(N,1)*frequencies-phases)
	trueW = np.random.randn(d,q)
	observed = np.dot(latents,trueW.T) + np.random.randn(N,d)*truesigma
	
	#PCA without missing values
	a = PCA_EM(observed,q)
	a.learn(niters)
	
	#a missing data problem
	Nmissing = int(N*missing_pc/100)
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
	pylab.plot(b.Xreconstruct)

	pylab.show()

