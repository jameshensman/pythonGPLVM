# -*- coding: utf-8 -*-
# Copyright 2009 James Hensman
# Licensed under the Gnu General Public license, see COPYING
import numpy as np
def checkgrad(f,fprime,x,step=1e-6, tolerance = 1e-4, *args):
	"""check the gradient function fprime by comparing it to a numerical estiamte from the function f"""
	
	#choose a random direction to step in:
	dx = step*np.sign(np.random.uniform(-1,1,x.shape))
	
	#evaulate around the point x
	f1 = f(x+dx,*args)
	f2 = f(x-dx,*args)
	
	numerical_gradient = (f1-f2)/(2*dx)
	gradient = fprime(x,*args)
	ratio = (f1-f2)/(2*np.dot(dx,gradient))
	print "gradient = ",gradient
	print "numerical gradient = ",numerical_gradient
	print "ratio = ", ratio, '\n'
	
	if np.abs(1-ratio)>tolerance:
		print "Ratio far from unity. Testing individual gradients"
		for i in range(len(x)):
			dx = np.zeros(x.shape)
			dx[i] = step*np.sign(np.random.uniform(-1,1,x[i].shape))
			
			f1 = f(x+dx,*args)
			f2 = f(x-dx,*args)
			
			numerical_gradient = (f1-f2)/(2*dx)
			gradient = fprime(x,*args)
			print i,"th element"
			#print "gradient = ",gradient
			#print "numerical gradient = ",numerical_gradient
			ratio = (f1-f2)/(2*np.dot(dx,gradient))
			print "ratio = ",ratio,'\n'
			
	