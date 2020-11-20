# SAMoSA: Read parameters from a json file. Made for CAPMD
import json


class Param:
	def __init__(self,filename):		
		self.filename=filename
		# magic to read in json file here
		# Assume it lives in paramdict, a python dictionary
		# change if necessary

		# here is the labels we definitely need:
		self.N = int(paramdict['N']) # Number of particles
		self.dt = float(paramdict['stepSize']) # time step
		self.dumpname = 'ParticleData'  # root name of particle data csv files

		# Geometry: always a plane, either with or without periodic boundary conditions
		if paramdict["periodic"] == '0':
			self.periodic = False
			self.constraint = 'plane'
		else:
			self.periodic = True
			self.constraint = 'plane_periodic'
		Lx = float(paramdict['L_x']) 
		Ly = float(paramdict['L_y']) 
		self.lx = Lx
		self.ly = Ly
		# put into a box to be compatible with samos convention
		# Note height. Just important it's larger than the maximum distance between contacting cells
		# Else problems with the cell list later on.
		self.box = [Lx,Ly,5.0]


		# Potential parameters (will need to be modified for several stiffnesses)
		self.potential = 'soft_attractive' # our current potential
		self.pot_params = {}
		self.pot_params['k'] = float(paramdict['kHarmonic']) # stiffness
		self.pot_params['re_fact'] = 1.15 # point at which the potential hits its minimum. We label that 1 + epsilon, I think?

		# alternative option when interactions are turned off
		self.potential = 'none'
		self.pot_params['k'] = 0.0

		# We are always using brownian dynamics 
		self.one_integrator = True
		self.integrator = 'brownian'
		self.seed =  # seed of orientational noise term
		self.mu = 1.0 # mobility (1/friction), usually 1 in simulation units
		self.v0 =  # self-propulsion speed
		self.nu = # magnitude of orientational noise
		# We may want to add alignment at some point.
		# Currently none
		self.J = 0.0
		self.aligner = 'none'
		# Velocity alignment
		#self.aligner = 'ajpolar'
		#self.J = float(paramdict["zetaVelocity"])

		# number of types in the simulation (including boundary)
		self.ntypes = 2

		# Still under construction: Dealing with different populations and their different division / death rates
		self.npopulation = 2 # number of different groups of cells doing their own division / death dynamics (e.g. = 2 for p53 / not p53 cells)
		# give them names by hand, can be part of json file too
		#self.population=['regular','p53']
		if self.npopulation>0:
			print ("Number of populations: " + str(self.npopulation))
			self.pop_params=[{} for k in range(self.npopulation)]
			for k in range(self.npopulation):
				self.population.append(conf.key_words['population'][k].name)
				self.pop_params[k]['divrate'] =  # division rate
				self.pop_params[k]['deathrate'] =  # death rate
				self.pop_params[k]['zmax'] =  # max contact number (for division, can have one for death too)
				self.pop_params[k]['frequency'] =  # after how many time steps we apply population in code
				# We can add others if and when necessary
			print ("Populations: ")
			print (self.population)
			print ("Population parameters: ")
			print (self.pop_params)

		# For our current code
		self.npopuation = 1
		self.population = ['regular']
		self.pop_params=[{}]
		self.pop_params[0]['divrate'] =  # division rate
		# etc
