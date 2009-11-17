# -*- coding: utf-8 -*-
import numpy as np
import pylab
from PCA_EM import PCA_EM
import kernels
import GP
from scipy import optimize
import MLP

class GPLVM:
	"""TODO: this should inherrit a GP, not contain an instance of it..."""
	def __init__(self,Y,dim):
		self.Xdim = dim
		self.N,self.Ydim = Y.shape
		
		"""Use PCA to initalise the problem. Uses EM version in this case..."""
		myPCA_EM = PCA_EM(Y,dim)
		myPCA_EM.learn(100)
		X = np.array(myPCA_EM.m_Z)
		
		self.GP = GP.GP(X,Y)#choose particular kernel here if so desired.
	
	def learn(self,niters):
		for i in range(niters):
			self.optimise_latents()
			self.optimise_GP_kernel()
			
	def optimise_GP_kernel(self):
		"""optimisation of the GP's kernel parameters"""
		self.GP.find_kernel_params()
		print self.GP.marginal(), 0.5*np.sum(np.square(self.GP.X))
	
	def ll(self,xx,i):
		self.GP.X[i] = xx
		self.GP.update()
		#return -self.GP.marginal()+ 0.5*np.sum(np.square(self.GP.X))#0.5*np.sum(np.square(s))
		return -self.GP.marginal()+ 0.5*np.sum(np.square(xx))
	
	def ll_grad(self,xx,i):
		self.GP.X[i] = xx
		self.GP.update()
		Kinv = np.linalg.solve(self.GP.L.T,np.linalg.solve(self.GP.L,np.eye(self.GP.L.shape[0])))
		
		matrix_grads = [self.GP.kernel.gradients_wrt_data(self.GP.X,i,jj) for jj in range(self.GP.Xdim)]
		
		alphalphK = np.dot(self.GP.A,self.GP.A.T)-self.GP.Ydim*Kinv
		grads = [-0.5*np.trace(np.dot(alphalphK,e)) for e in matrix_grads]
			
		return np.array(grads) + xx
		
	def optimise_latents(self):
		"""Direct optimisation of the latents variables."""
		xtemp = np.zeros(self.GP.X.shape)
		for i,yy in enumerate(self.GP.Y):
			original_x = self.GP.X[i].copy()
			#xopt = optimize.fmin(self.ll,self.GP.X[i],disp=True,args=(i,))
			xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,disp=True,args=(i,))
			self.GP.X[i] = original_x
			xtemp[i] = xopt
		self.GP.X = xtemp.copy()
		
		
		
	
class GPLVMC(GPLVM):
	"""A(back) constrained version of the GPLVM"""
	def __init__(self,data,xdim,nhidden=5,mlp_alpha=2):
		GPLVM.__init__(self,data,xdim)
		
		self.MLP = MLP.MLP((self.Ydim,nhidden,self.Xdim),alpha=mlp_alpha)
		self.MLP.train(self.GP.Y,self.GP.X)#create an MLP initialised to the PCA solution...
		self.GP.X = self.MLP.forward(self.GP.Y)
		#bypass the GP class's normalising stuff
		self.GP.xmean = 0
		self.GP.xstd = 1
		
	def ll(self,w):
		self.MLP.unpack(w[:-3])
		self.GP.X = self.MLP.forward(self.GP.Y)
		return  self.GP.ll(w[-3:]) + 0.5*np.sum(np.square(self.GP.X)) + 0.5*self.MLP.alpha*np.sum(np.square(w[:-3]))#regulariser on mlp weights TODO: move into MLP code
	
	
	def ll_grad(self,w):
		self.MLP.unpack(w[:-3])
		self.GP.X = self.MLP.forward(self.GP.Y)
		
		#gradients wrt the GP parameters can be done inside the GP class. This also sets the kernel parameters, computes alphalphK.
		GP_grads = self.GP.ll_grad(w[-3:])
		
		#gradient matrices (gradients of the kernel matrix wrt data)
		gradient_matrices = self.GP.kernel.gradients_wrt_data(self.GP.X)
		#gradients of the error function wrt 'network outputs', i.e. latent variables
		x_gradients = np.array([-0.5*np.trace(np.dot(self.GP.alphalphK,e)) for e in gradient_matrices]).reshape(self.GP.X.shape) + self.GP.X
		
		weight_gradients = self.MLP.backpropagate(self.GP.Y,x_gradients) + self.MLP.alpha*w[:-3]
		
		return np.hstack((weight_gradients,GP_grads))
		
	def learn(self,callback=None):
		"""'Learn' by optimising the weights of the MLP and the GP hyper parameters together.  """
		#w_opt = optimize.fmin(self.ll,np.hstack((self.MLP.pack(),self.GP.kernel.get_params(),np.log(self.GP.beta))),maxiter=miter,maxfun=meval)
		w_opt = optimize.fmin_cg(self.ll,np.hstack((self.MLP.pack(),self.GP.kernel.get_params(),np.log(self.GP.beta))),self.ll_grad,args=(),callback=callback)
		final_cost = self.ll(w_opt)#sets all the parameters...
		


if __name__=="__main__":
	N = 50
	colours = np.arange(N)#something to colour the dots with...
	theta = np.linspace(2,6,N)
	Y = np.vstack((np.sin(theta)*(1+theta),np.cos(theta)*theta)).T
	Y += 0.1*np.random.randn(N,2)
	
	thetanorm = (theta-theta.mean())/theta.std()
	
	xlin = np.linspace(-1,1,1000).reshape(1000,1)
	
	myGPLVM = GPLVMC(Y,1,nhidden=3)
	
	def plot_current():
		pylab.figure()
		ax = pylab.axes([0.05,0.8,0.9,0.15])
		pylab.scatter(myGPLVM.GP.X[:,0]/myGPLVM.GP.X.std(),np.zeros(N),40,colours)
		pylab.scatter(thetanorm,np.ones(N)/2,40,colours)
		pylab.yticks([]);pylab.ylim(-0.5,1)
		ax = pylab.axes([0.05,0.05,0.9,0.7])
		pylab.scatter(Y[:,0],Y[:,1],40,colours)
		Y_pred = myGPLVM.GP.predict(xlin)[0]
		pylab.plot(Y_pred[:,0],Y_pred[:,1],'b')
	
	class callback:
		def __init__(self,print_interval,objective_fn):
			self.counter = 0
			self.print_interval = print_interval
			self.objective_fn = objective_fn
		def __call__(self,w):
			self.counter +=1
			if not self.counter%self.print_interval:
				print self.counter, 'iterations, cost: ',self.objective_fn(w)
	cb = callback(100,myGPLVM.ll)
			
	#plot_current()
	myGPLVM.learn(callback=cb)
	plot_current()
	
	pylab.show()
		
		
		
		
