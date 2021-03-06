

from read_param import *
import numpy as np

# Given importance of radii et al, we have to pass Interaction the information about
# radii (and positions)? Alternatively, pass it just the unchanging particle labels
class Interaction:

	def __init__(self,param,sigma,ignore=False,debug=False):
		self.param=param
		self.sigma= sigma
		self.ignore=ignore
		self.debug=debug
		print(self.param.ntypes)
		print(self.ignore)
		# First step: unpack parameters to find out what kind of types we have, and what kind of potentials
		# Ignore is for cornea or crowds, where despite different types, they all have the same stiffnesses
		if ((self.param.ntypes==1) or (self.ignore)):
			if self.param.potential=='soft':
				self.k=self.param.pot_params['k']
				self.dmax=2*self.sigma
				self.mult=1
			elif self.param.potential=='soft_attractive':
				self.k=self.param.pot_params['k']
				self.fact=self.param.pot_params['re_fact']-1.0
				self.rmax=1+2.0*self.fact
				self.dmax=2*self.sigma
				self.mult=self.rmax
			elif self.param.potential=='morse':
				self.D=self.param.pot_params['D']
				self.re=self.param.pot_params['re']
				self.a=self.param.pot.params['a']
				self.dmax=4*self.sigma
				self.mult=self.re
			elif self.param.potential=='gaussian':
				# give it at least a default neighbour radius ...
				self.dmax=2*self.sigma
				print("Warning! Gaussian interaction has not yet been implemented!") 
			elif self.param.potential=='rod':
				# CHANGE: Assuming here rods are at default aspect ratio 5
				self.dmax=10*self.sigma
				self.mult=1.0
				print("Warning! Rod interaction has not yet been implemented!")
			else:
				# give it at least a default neighbour radius ...
				self.dmax=2*self.sigma
				self.mult=1.0
				#print "Warning! Unknown interaction type! "
		else:
			# give it at least a default neighbour radius ...
			self.dmax=2*self.sigma
			self.mult=1.0
			print("Warning! Multiple types of particles interacting have not yet been implemented! Defaulting to monodisperse ofr radius 1")
		
	def getDmax(self):
		return self.dmax
	
	def getMult(self):
		return self.mult
		
	# also is -gradient potential
	# Manually remove issues of particles on top of each other. Cause is usually sloppy implementation of glued boundaries.
	def getForce(self,i,neighbours,drvec,radi,radj,eps=1e-8):
		Fvec=0.0*(np.array(neighbours).transpose()*(drvec).transpose()).transpose()
		dr=np.sqrt(drvec[:,0]**2+drvec[:,1]**2+drvec[:,2]**2)
		ontop = [index for index, value in enumerate(dr) if value < eps]
		if ((self.param.ntypes==1) or (self.ignore==True)):
			if self.param.potential=='soft':	
				diff=radi+radj-dr
				fact = 0.5*self.k*diff
				Fvec=self.k*((diff/dr).transpose()*(drvec).transpose()).transpose()
			elif self.param.potential=='soft_attractive':
				scale=radi+radj
				diff=scale-dr
				dscaled=diff/scale
				rep = [index for index, value in enumerate(dscaled) if value > -self.fact]
				#print "Before upshot ones: " + str(len(rep))
				att = [index for index, value in enumerate(dscaled) if value <= -self.fact]
				#print "Attractive after upshot ones: " + str(len(att))
				factor=np.empty((len(neighbours),))
				# repulsive ones
				factor[rep] = self.k*diff[rep]
				# attractive ones
				factor[att]=-self.k*(self.rmax*scale[att]-dr[att])
				Fvec=((factor/dr).transpose()*(drvec).transpose()).transpose()
			elif self.param.potential=='morse':
				fnorm=-2*self.a*self.D*np.exp(-self.a*(dr-self.re))*(1-np.exp(-self.a*(dr-self.re)))
				Fvec=((fnorm/dr).transpose()*(drvec).transpose()).transpose()
			elif self.param.potential=='gaussian':
				print("Warning! Gaussian interaction has not yet been implemented! Returning zero force")
			elif self.param.potential=='rod':
				print("Warning! Rod interaction has not yet been implemented! Returning zero force")
			else:
				print("Warning! Unknown interaction type! Returning zero force")
		else:
			# Do the Morse right now only ... will serve as a template
			print("Warning! Multiple types of particles interacting have not yet been implemented! Returning zero force")
		if len(ontop)>0:
			print("Warning: Found particles on top of each other. Setting force to 0 here: ")
			print (i,neighbours[ontop[0]])
			for o in ontop:
				Fvec[o,:]=np.zeros((3,))
			print(Fvec)
		return Fvec
				
	def getStresses(self,i,neighbours,drvec,radi,radj):
		# Do these all the standard way, including the pressure as trace of the matrix
		stress=np.zeros((len(neighbours),3,3))
		# First get the forces:
		Fvec=self.getForce(i,neighbours,drvec,radi,radj)
		for u in range(3):
			for v in range(3):
				stress[:,u,v]+=0.5*drvec[:,u]*Fvec[:,v]
		# Then get the pressure as the trace over the last two components
		press=np.trace(stress,axis1=1,axis2=2)
		return press,stress
	  
	def getEnergy(self,i,neighbours,drvec,radi,radj):
		# Note: There is a 0.5 before the energy return statements because as written, every contact is counted twice in the calculation
		dr=np.sqrt(drvec[:,0]**2+drvec[:,1]**2+drvec[:,2]**2)
		if ((self.param.ntypes==1) or (self.ignore==True)):
			if self.param.potential=='soft':	
				diff=radi+radj-dr
				fact = 0.5*self.k*diff
				eng_val = fact*diff
				return 0.5*eng_val
			elif self.param.potential=='soft_attractive':
				scale=radi+radj
				diff=scale-dr
				dscaled=diff/scale
				rep = [index for index, value in enumerate(dscaled) if value > -self.fact]
				#print "Before upshot ones: " + str(len(rep))
				att = [index for index, value in enumerate(dscaled) if value <= -self.fact]
				#print "Attractive after upshot ones: " + str(len(att))
				factor=np.empty((len(neighbours),))
				# repulsive ones
				factor[rep] = self.k*diff[rep]
				# attractive ones
				factor[att]=-self.k*(self.rmax*scale[att]-dr[att])
				eng_val=np.empty((len(neighbours),))
				eng_val[rep] = 0.5*factor[rep]*diff[rep]
				eng0=0.5*self.k*(self.fact*scale[att])**2
				eng_val[att] = eng0+(eng0-(factor[att]**2)/self.k)
				return 0.5*eng_val
			elif self.param.potential=='morse':
				eng_val=self.D*(1-np.exp(-self.a*(dr-self.re)))**2
				return 0.5*eng_val
			elif self.param.potential=='gaussian':
				print("Warning! Gaussian interaction has not yet been implemented! Returning zero energy")
				return 0
			elif self.param.potential=='rod':
				print("Warning! Rod interaction has not yet been implemented! Returning zero energy")
				return 0
			else:
				#print "Warning! Unknown interaction type! Returning zero energy"
				return 0
		else:
			# Do the Morse right now only ... will serve as a template
			print("Warning! Multiple types of particles interacting have not yet been implemented! Returning zero energy")
			return 0
	  
	def getStiffness(self,i,neighbours,drvec,radi,radj):
		dr=np.sqrt(drvec[:,0]**2+drvec[:,1]**2+drvec[:,2]**2)
		if ((self.param.ntypes==1) or (self.ignore==True)):
			if self.param.potential=='soft':
				return self.k*np.ones((len(neighbours),))
			elif self.param.potential=='soft_attractive':
				scale=radi+radj
				diff=scale-dr
				dscaled=diff/scale
				rep = [index for index, value in enumerate(dscaled) if value > -self.fact]
				att = [index for index, value in enumerate(dscaled) if value <= -self.fact]
				stiff=np.zeros((len(neighbours),))
				stiff[rep]=self.k 
				stiff[att]=-self.k
				return stiff
			elif self.param.potential=='morse':
				stiff=2.0*self.a**2*self.D*np.exp(-self.a*(dr-self.re))*(2.0*np.exp(-self.a*(dr-self.re))-1)
				return stiff
			elif self.param.potential=='gaussian':
				print("Warning! Gaussian interaction has not yet been implemented! Returning zero stiffness")
				return np.zeros((len(neighbours),))
			elif self.param.potential=='rod':
				print("Warning! Rod interaction has not yet been implemented! Returning zero stiffness")
				return np.zeros((len(neighbours),))
			else:
				#print "Warning! Unknown interaction type! Returning zero stiffness"
				return np.zeros((len(neighbours),))
		else:
			# Do the Morse right now only ... will serve as a template
			print("Warning! Multiple types of particles interacting have not yet been implemented! Returning zero stiffness")
