# SAMoSA: Read parameters from a keyword, value type csv file. Made for CCCPy.

import csv


class Param:
	def __init__(self,filename):		
		self.filename=filename

		with open(self.filename, newline='') as csvfile:
			# First read in the keyword: value styled parameter file
			paramreader = csv.reader(csvfile, delimiter=',')
			paramdict = {rows[0]:rows[1] for rows in paramreader}
			print(paramdict)

			# the construct bespoke parameters out of it
			self.N = int(paramdict['N'])
			self.dt = float(paramdict['stepSize'])
			self.dumpname = 'ParticleData'

			# Geometry
			if paramdict["periodic"] == '0':
				self.periodic = False
				self.constraint = 'plane'
			else:
				self.periodic = True
				self.constraint = 'plane_periodic'
			Lx = float(paramdict['L_x']) 
			Ly = float(paramdict['L_y']) 
			# This box is too tight?
			if not self.periodic:
				Lx = Lx + 5
				Ly = Ly + 5
			# put into a box to be compatible with samos convention
			# Note height ("corridor?"). Just important it's not zero
			# Has to be larger than rcut
			self.box = [Lx,Ly,5.0]
			# and because that appears to be used as well ... cleanup
			self.lx = Lx
			self.ly = Ly

			self.ntypes = 2

			# elastic potential 
			self.pot_params = {}
			if paramdict["enableHarmonicInterForce"]=='1':
				self.potential = 'soft'
				self.pot_params['k'] = float(paramdict['kHarmonic'])
			elif paramdict["enableHertzianInterForce"]=='1':
				self.potential = 'hertzian'
				self.pot_params['k'] = float(paramdict['kHertzian'])
			else:
				print("No or unknown particle interactions")
				self.potential = 'none'
				self.pot_params['k'] = 0.0
				
			# We know what this is
			self.one_integrator = True
			self.integrator = 'langevin'

			# Dynamical parameters (as in 'oneInt' of Samos)
			if paramdict["enableActiveForce"]=='1':
				self.v0 = paramdict["v_0"]
			else:
				self.v0 = 0.0
				
			# angular noise
			if paramdict["enableRandNoisyTorque"]=='1':
				self.nu = float(paramdict['sigmaTorque'])
			else:
				self.nu = 0.0
				
			# mobility = 1 / friction coefficient
			# Note that we are currently setting ground and active friction to the same value
			# Revisit assumption if necessary
			if paramdict["enableGroundFrictionForce"] == '1':
				try:
					self.mu = 1.0/paramdict["zetaGround"]
				except:
					self.mu = 1.0
				#zetaActive,0.100000
				#zetaGround,0.100000
				self.zeta = paramdict["zetaActive"]
			else:
				self.zeta = 0.0 
				
			# Spatial noise value (aka temperature, but hold off)
			# Note that there are different x and y values considered here ...
			if paramdict["enableRandNoisyForce"] == '1':
				self.nu_spatial = float(paramdict["sigmaForceX"])
			else:
				self.nu_spatial = 0.0
			
			# We also have angular dissipation
			if paramdict["enableAngularFrictionTorque"] == '1':
				self.zeta_angle = paramdict["xiAngular"]
			else:
				self.zeta_angle = 0.0
				
			# Also get alignment parameters
			# Currently under the assumption that it's *either* polar alignment *or* velocity alignment, but not both
			self.J = 0.0
			self.aligner = 'none'
			if paramdict["enableVelocityAlignmentTorque"] == '1':
				self.aligner = 'ajpolar'
				self.J = float(paramdict["zetaVelocity"])
			elif paramdict["enablePolarAlignmentTorque"] == '1':
				self.aligner = 'polar'
				self.J = float(paramdict["zetaPolar"])
				
			# There are a whole bunch of other parameters here which do not exist for brownian dynamics
			self.mass_radius_ratio = float(paramdict["massRadiusRatio"])
			
			# pair friction force
			if paramdict["enablePersonFrictionForce"] == '1':
				self.zeta_pair = float(paramdict["zetaPerson"])
			else:
				self.zeta_pair = 0.0
				
			# pair friction torque 
			if paramdict["enablePairDissipationTorque"] == '1':
				self.zeta_pair_angle = float(paramdict["xiPair"])
			else:
				self.zeta_pair_angle = 0.0
			
			
			
				
			
				
				
			
			
				
				




