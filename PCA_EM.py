# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
from numpy import matlib as ml
import numpy as np

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
		M_inv = np.linalg.inv(M)
		self.m_Z = np.dot(M_inv,np.dot(self.W.T,self.X2.T)).T
		self.S_z = M_inv*self.sigma2
	def M_step(self):
		zzT = np.dot(self.m_Z.T,self.m_Z) + self.N*self.S_z
		self.W = np.dot(np.dot(self.X2.T,self.m_Z),np.linalg.inv(zzT))
		WTW = np.dot(self.W.T,self.W)
		self.sigma2 = self.xxTsum - 2*np.sum(np.dot(self.m_Z,self.W.T)*self.X2) + np.trace(np.dot(zzT,WTW))
		#self.sigma2 = self.xxTsum - 2*ml.trace(self.m_Z*self.W.T*self.X2.T) + ml.trace(zzT*WTW)
		#self.sigma2 = self.xxTsum + ml.sum([- 2*z*self.W.T*x.T + ml.trace((z.T*z + self.S_z)*WTW) for z,x in zip(self.m_Z, self.X2)])
		self.sigma2 /= self.N*self.d
		

if __name__=='__main__':
	q=2
	d=50
	N=500
	truesigma = 4.
	latents = ml.randn(N,q)
	trueW = ml.randn(d,q)
	observed = latents*trueW.T + ml.randn(N,d)*truesigma
	a = PCA_EM(observed,q)
	a.learn(500)
	from hinton import hinton
	import pylab
	hinton(ml.linalg.qr(trueW.T)[1].T)
	pylab.figure()
	hinton(ml.linalg.qr(a.W.T)[1].T)
	pylab.figure()
	pylab.scatter(ml.asarray(latents)[:,0],ml.asarray(latents)[:,1],40,ml.asarray(latents)[:,0])
	pylab.figure()
	pylab.scatter(ml.asarray(a.m_Z)[:,0],ml.asarray(a.m_Z)[:,1],40,ml.asarray(latents)[:,0])

	#pylab.show()

