
from Geometry import *
#from Tesselation import *
#from Defects import *
#from Dynamics import *
from Interaction import *

from read_param import *
from read_data import *
from CellList import *

import sys


try:
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	HAS_MATPLOTLIB=True
except:
	HAS_MATPLOTLIB=False
	pass

# Global list of exisiting geometries
geometries={'sphere':GeometrySphere,'plane':GeometryPlane,'plane_periodic':GeometryPeriodicPlane,'none':Geometry,'tube':GeometryTube,'peanut':GeometryPeanut,'hourglass':GeometryHourglass}

class Configuration:
	# We need a completely generic constructor since we will have to make sub-classes as well
	# as well as some with fresh positions in python from other code
	def __init__(self,**kwargs):
		print ' kwargs: ', kwargs
		self.initype = kwargs["initype"]
		if self.initype == "fromCSV":
			self.fromCSV(kwargs["parampath"],kwargs["datapath"],kwargs["multiopt"])
		elif self.initype == "makeChild":
			self.makeChild(kwargs["parentconf"],kwargs["frame"],kwargs["usetype"])
		elif self.initype == "fromPython":
			self.fromPython(kwargs["param"],kwargs["rval"],kwargs["vval"],kwargs["nval"],kwargs["radii"],kwargs["ptype"],kwargs["flag"])
		else:
			print("Configuration:Unknown configuration constructor option, stopping!")
			break
			
			
	# Use methods for the actual initialisation
	def fromCSV(self,parampath,datapath,multiopt,debug=False):
		self.debug = debug
		self.param = Param(parampath)
		self.datapath = datapath
		self.multiopt = multiopt
		# Options for running mode the analysis
		if self.multiopt == "single":
			print("Configuring to read single files")
		elif self.multiopt == "many":
			print("Configuring to work on many files (glassy dynamics)")
		else:
			print("Unknown file mode configuration option, stoppping!")
			break
		# Create the right geometry environment:
		self.geom=geometries[param.constraint](param)
		print self.geom
		
	# Create configuration from some python data passed down to here
	def fromPython(self,param,rval,vval,nval,radius,ptype,flag):
		self.debug = False
		self.param = param
		self.multiopt = "single"
		self.usetype = "all"
		
		self.rval = rval
		self.vval = vval
		self.nval = nval
		self.radius = radius
		self.ptype = ptype
		self.flag = flag
		
		self.N = len(radius)
		self.sigma = np.mean(radius)
		
		self.inter=Interaction(self.param,False,self.radius,False)
		self.geom=geometries[param.constraint](param)
		
		
				
	# Glorified copy constructor to isolate one frame and a subset of particles. Will serve as input to Hessian or Tesselation
	def makeChild(self,parentconf,frame=1,usetype='all'):
		self.debug = parentconf.debug
		self.param = parentconf.param
		# Single frame for the child
		self.multiopt = "single"
		# And we have already narrowed down to the types we are looking for
		self.usetype = "all"
		self.sigma = parentconf.sigma
		
		if parentconf.multiopt=="single":
			useparts = parentconf.getUseparts(usetype)
			self.N = len(useparts)
			self.rval = parentconf.rval[useparts,:]
			self.vval = parentconf.vval[useparts,:]
			self.nval = parentconf.nval[useparts,:]
			self.radius = parentconf.radius[useparts]
			self.ptype = parentconf.ptype[useparts]
			self.flag = parentconf.flag[useparts]
		else:
			useparts = parentconf.getUseparts(usetype,frame)
			self.N = len(useparts)
			self.rval = parentconf.rval[frame,useparts,:]
			self.vval = parentconf.vval[frame,useparts,:]
			self.nval = parentconf.nval[frame,useparts,:]
			self.radius = parentconf.radius[frame,useparts]
			self.ptype = parentconf.ptype[frame,useparts]
			self.flag = parentconf.flag[frame,useparts]
			
		# is that passing a pointer? Or should I make some new ones instead?
		#self.inter=Interaction(self.param,monodisperse,self.radius,self.ignore)
		#self.geom=geometries[param.constraint](param)
		self.inter = parentconf.inter
		self.geom = parentconf.geom
		
		
	def readDataSingle(self,filename,internal=False,readtypes='all'):
		print("Processing file : " + filename)
		data = ReadData(filename)
		
		# Positions and velocities: non-negotiable in a data set
		x, y, z = np.array(data.data[data.keys['x']]), np.array(data.data[data.keys['y']]), np.array(data.data[data.keys['z']])
		vx, vy, vz = np.array(data.data[data.keys['vx']]), np.array(data.data[data.keys['vy']]), np.array(data.data[data.keys['vz']])
		
		if data.keys.has_key('type'):
			ptype0 = data.data[data.keys['type']]
			if readtypes == 'all':
				N = len(x)
				ptype = ptype0
			else:
				useparts=[]
				for tp in ptype:
					if tp in readtypes:
						useparts.append(tp)
				if len(useparts)==0:
					print('Error: No particles of the correct types in simulation, looking for ' + str(readtypes))
					break
				else:
					N=len(useparts)
					ptype = ptype0[useparts]
		else:
			if readtypes == 'all':
				N = len(x)
				ptype = np.ones((N,))
				useparts = range(N)
			else:
				print("Error: looking for data with types " + str(readtypes) + " but data has no type information.")
				break
			
		
		rval = np.column_stack((x[useparts],y[useparts],z[useparts]))
		vval = np.column_stack((vx[useparts],vy[useparts],vz[useparts]))
		
		# Attempt to read active force direction, else this is all 0
		try:
			nx, ny, nz = np.array(data.data[data.keys['nx']]), np.array(data.data[data.keys['ny']]), np.array(data.data[data.keys['nz']])
		except KeyError:
			nx, ny, nz = np.zeros(np.shape(x)),np.zeros(np.shape(y)),np.zeros(np.shape(z))
		nval = np.column_stack((nx[useparts],ny[useparts],nz[useparts]))
		
		# Attempt to read radius, type and flag if present
		# Set spatial scale sigma for the cell list builder
		
		if not data.keys.has_key('radius'): 
			radius = np.ones(N)
			monodisperse=True
			sigma = 1.0
		else: 
			radius0 = np.array(data.data[data.keys['radius']])	
			radius = radius0[useparts]
			sigma = np.mean(radius)
		
		if data.keys.has_key('flag'):
			flag0 = data.data[data.keys['flag']]
			flag = flag0[useparts]
		else:
			flag = range(N)
			
		# Do the rest of the configuration only if this is not part of a series of reading in data
		if internal:
			return N,rval,vval,nval,radius,ptype,flag,sigma,monodisperse
		else:
			self.N = N
			self.rval = rval
			self.vval = vval
			self.nval = nval
			self.radius = radius
			self.sigma = sigma
			self.ptype = ptype
			self.flag = flag
		
			# Create the Interaction class
			self.inter=Interaction(self.param,monodisperse,self.radius,self.ignore)
			
			# Apply periodic geomtry conditions just in case (there seem to be some rounding errors floating around)
			if self.geom.periodic:
				self.rval=self.geom.ApplyPeriodic2d(self.rval)
			
			if self.debug:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(self.rval[:,0], self.rval[:,1], self.rval[:,2], zdir='z', c='b')
			return 0
		
	
	# read in data based on the content of the folder
	# readtypes is now a list (when it is not 'all'): Read data of these types, which contains the tracer options
	# tracers is type = [2], cornea is types = [1, 2, 3] 
	def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
		self.Nvariable=Nvariable
		try:
			filepattern = self.param.dumpname
		except:
			filepattern = 'frame'
		filepattern = self.datapath + namepatt
		files0 = sorted(glob(self.datapath + filepattern+'*.dat'))
		if len(files0) == 0:
  			files0 = sorted(glob(self.datapath + filepattern+'*.dat.gz'))
		if len(files0) == 0:
			print("Error: could not identify data file name pattern, does not end in .dat or .dat.gz, or is not part of parameters, and is not 'frame'. Stopping.")
			break
		
		if howmany == 'all':
			files = files0
		else:
			nfin = skip + step*howmany
			if nfin>len(files0):
				nfin = len(files0)
				print("Warning: not enough files in simulation, reading only " + str((nfin-skip)/step) + " files.")
			files = files0[skip:nfin:step]
		self.Nsnap=len(files)
		
		# If we know that the number of particles is variable
		# We need to actually go read the number of particles in each file first
		if self.Nvariable:
			self.Nval=np.zeros((self.Nsnap,))
			u=0
			for f in files:
				#print "Pre - Processing file : ", f
				data = ReadData(f)
				x= np.array(data.data[data.keys['x']])
				if readtypes = 'all':
					self.Nval[u]=len(x)
				else:
					if data.keys.has_key('type'):
						ptype = data.data[data.keys['type']]
						for tp in ptype:
							if tp in readtypes:
								useparts.append(tp)
						if len(useparts)==0:
							print('Error: No particles of the correct types in simulation, looking for ' + str(readtypes))
							break
						else:
							self.Nval[u]=len(useparts)
					else:
						print("Error: looking for data with types " + str(readtypes) + " but data has no type information.")
						break
				#print self.Nval[u]	
				u+=1
			self.N=int(np.amax(self.Nval))
		else:
			self.N,rval,vval,nval,radius,ptype,flag,self.sigma,monodisperse = self.readDataSingle(files[0],True,readtypes)
			self.Nval=[self.N for i in range(self.Nsnap)]
		print("Handling a total of maximum " + str(self.N) + " particles!")
			
		
		self.rval=np.zeros((self.Nsnap,self.N,3))
		self.vval=np.zeros((self.Nsnap,self.N,3))
		self.nval=np.zeros((self.Nsnap,self.N,3))
		self.flag=np.zeros((self.Nsnap,self.N))
		self.radius=np.zeros((self.Nsnap,self.N))
		self.ptype=np.zeros((self.Nsnap,self.N))
		u=0
		for f in files:
			# first read the data, for all types
			N,rval,vval,nval,radius,ptype,flag,sigma,monodisperse = self.readDataSingle(f,True,readtypes)
			# Running tab on maximum particle size for CellList
			if sigma>self.sigma
				self.sigma = sigma
			# Then arrange in a sensible shape
			self.rval[u,:self.Nval(u),:]=rval
			self.vval[u,:self.Nval(u),:]=vval
			self.nval[u,:self.Nval(u),:]=nval
			self.flag[u,:self.Nval(u)]=flag
			self.radius[u,:self.Nval(u)]=radius
			self.ptype[u,:self.Nval(u)]=ptype
		
		# Create the Interaction class
		self.inter=Interaction(self.param,monodisperse,self.radius,self.ignore)
			
		# Apply periodic geomtry conditions just in case 
		if self.geom.periodic:
			self.rval=self.geom.ApplyPeriodic3d(self.rval)
		
	# utility function to get the partcles we want
	# Note: this can be a further subset of the ones we have read in, e.g. for tracer particles
	# including the 'do nothing' option for speed and convenience
	def getUseparts(self,usetype='all',frame=1):
		if usetype == 'all':
			if self.multiopt=="single":
				return range(self.N)
			else:
				return range(self.Nval[frame])
		else:
			if self.multiopt=="single":
				ptype = self.ptype
			else:
				ptype = self.ptype[frame,:self.Nval[frame]]
			useparts=[]
			for tp in ptype:
				if tp in usetype:
					useparts.append(tp)
			if len(useparts)==0:
				print("Configuration::getUseparts - Warning: no particles of the desired type(s) " + str(usetype))
			return useparts
			
		
					
	# create Cell list, but only ever for a single frame (?)
	def makeCellList(self,frame=1):
		# Delete any existing cell list
		try:
			del self.clist
		except:
			pass
		# Create the cell list, based on either neighbour list cutoff if defined, or else mean radius
		try:
			cellsize=param.nlist_rcut
		except:
			cellsize = 3*self.sigma
		if cellsize>5*self.sigma:
			cellsize=5*self.sigma
			print("Warning! Reduced the cell size to manageable proportions (5 times mean radius). Re-check if simulating very long objects!")
		self.clist=CellList(self.geom,cellsize)
		# Populate it with all the particles:
		if self.multiopt=="single"
			for k in range(self.N):
				self.clist.add_particle(self.rval[k,:],k)
				
		else:
			for k in range(self.Nval[frame]):
				self.clist.add_particle(self.rval[frame,k,:],k)
		if self.debug:
			self.clist.printMe()	
		
		
	# Tangent bundle: Coordinates in an appropriate coordinate system defined on the manifold
	# and the coordinate vectors themselves, for all the particles
	def getTangentBundle(self,frame=1):
		if self.multiopt=="single"
			self.x1,self.x2,self.e1,self.e2=self.geom.TangentBundle(self.rval)
			return self.x1,self.x2,self.e1,self.e2
		else:
			self.x1,self.x2,self.e1,self.e2=self.geom.TangentBundle(self.rval[frame,self.Nval[frame],:])
			return self.x1,self.x2,self.e1,self.e2
	
	# Rotate the frame (for bands on spheres). If in multimodus, indicate frame as well.
	def rotateFrame(self,axis,rot_angle,frame=1):
		if self.multiopt=="single"
			self.rval = self.geom.RotateVectorial(self.rval,axis,-rot_angle)
			self.vval = self.geom.RotateVectorial(self.vval,axis,-rot_angle)
			self.nval = self.geom.RotateVectorial(self.nval,axis,-rot_angle)
			self.nval=((self.nval).transpose()/(np.sqrt(np.sum(self.nval**2,axis=1))).transpose()).transpose()
			#self.vel = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2 + self.vval[:,2]**2)
			# redo the cell list
			self.makeCellList()
		else:
			rval0 = self.geom.RotateVectorial(self.rval[frame,self.Nval[frame],:],axis,-rot_angle)
			vval0 = self.geom.RotateVectorial(self.vval[frame,self.Nval[frame],:],axis,-rot_angle)
			nval0 = self.geom.RotateVectorial(self.nval[frame,self.Nval[frame],:],axis,-rot_angle)
			nval0=((nval0).transpose()/(np.sqrt(np.sum(nval0**2,axis=1))).transpose()).transpose()
			self.rval[frame,self.Nval[frame],:]=rval0
			self.vval[frame,self.Nval[frame],:]=vval0
			self.nval[frame,self.Nval[frame],:]=nval0
			# redo the cell list
			self.makeCellList(frame)
			
	# get neighbours of a particle
	def getNeighbours(self,i,mult=1.0,dmax=2*self.sigma,frame=1):
		# Find potential neighbours from neighbour list first
		if self.multiopt=="single":
			cneighbours=self.clist.get_neighbours(self.rval[i,:])
			#print "Cell list neighbours: " + str(len(cneighbours))
			drvec0=self.geom.ApplyPeriodic2d(self.rval[cneighbours,:]-self.rval[i,:])
		else:
			cneighbours=self.clist.get_neighbours(self.rval[frame,i,:])
			#print "Cell list neighbours: " + str(len(cneighbours))
			drvec0=self.geom.ApplyPeriodic2d(self.rval[frame,cneighbours,:]-self.rval[frame,i,:])
		dist=np.sqrt(drvec0[:,0]**2+drvec0[:,1]**2+drvec0[:,2]**2)
		#dist=self.geom.GeodesicDistance12(self.rval[cneighbours,:],self.rval[i,:])
		#print "Mean cutoff: " + str(mult*dmax)
		if self.monodisperse: 
			neighbours=[cneighbours[index] for index,value in enumerate(dist) if value <mult*dmax]
			radi = self.sigma
			radj = self.sigma*np.ones(len(neighbours))
		else:
			if self.multiopt=="single":
				neighbours=[cneighbours[index] for index,value in enumerate(dist) if value < mult*(self.radius[i]+self.radius[cneighbours[index]])]
				radi = self.radius[i]
				radj = self.radius[neighbours]
			else:
				neighbours=[cneighbours[index] for index,value in enumerate(dist) if value < mult*(self.radius[frame,i]+self.radius[frame,cneighbours[index]])]
				radi = self.radius[frame,i]
				radj = self.radius[frame,neighbours]
		## Stupid one for debugging purposes:
		#dist=self.geom.GeodesicDistance12(self.rval,self.rval[i,:])
		#neighbours = [index for index, value in enumerate(dist) if value < mult*(self.radius[i]+self.radius[index])]
		neighbours.remove(i)
		#print "Contact neighbours: " + str(len(neighbours))
		#print neighbours
		if self.multiopt=="single":
			drvec=self.geom.ApplyPeriodic2d(self.rval[neighbours,:]-self.rval[i,:])
		else:
			drvec=self.geom.ApplyPeriodic2d(self.rval[frame,neighbours,:]-self.rval[frame,i,:])
		#dr=np.sqrt(drvec[:,0]**2+drvec[:,1]**2+drvec[:,2]**2)
		return neighbours, drvec, radi, radj
	      
	def compute_energy_and_pressure(self,frame=1):
		if self.multiopt--"single":
			N = self.N
		else:
			N = self.Nval[frame]
		eng = np.zeros(N)
		press = np.zeros(N)
		ncon = np.zeros(N)
		stress = np.zeros((N,3,3))
		for i in range(N):
			neighbours, drvec, radi, radj=self.getNeighbours(i,self.inter.getMult(),self.inter.getDmax(),frame)
			ncon[i]=len(neighbours)
			eng[neighbours]+=self.inter.getEnergy(i,neighbours,drvec,radi,radj)
			press_val,stress_val=self.inter.getStresses(i,neighbours,drvec,radi,radj)
			stress[neighbours,:,:]+=stress_val
			press[neighbours]+=press_val
		return [eng, press, ncon,stress]
	
	
	# Basic statistics (mean velocity, density, pressure, stress)
	def getStatsBasic(self,frame=1):
		if self.multiopt=="single":
			vel2 = self.vval[:,0]**2 + self.vval[:,1]**2 + self.vval[:,2]**2
			phival=np.pi*np.sum(self.radius**2)/self.geom.area
			ndensity=self.N/self.geom.area
			eng, press,ncon,stress = self.compute_energy_and_pressure()
		else:
			vel2 = self.vval[frame,:,0]**2 + self.vval[frame,:,1]**2 + self.vval[frame,:,2]**2
			phival=np.pi*np.sum(self.radius[frame,:]**2)/self.geom.area
			ndensity=self.Nval[frame]/self.geom.area
			eng, press,ncon,stress = self.compute_energy_and_pressure(frame)
		vel2av=np.mean(vel2)
		pressure=np.sum(press)/self.geom.area
		fmoment=np.mean(press)
		energy=np.mean(eng)
		energytot=np.sum(eng)
		zav=np.mean(ncon)
		return vel2av, phival,ndensity, pressure,fmoment,energy,energytot,zav
	
	
			
	
	
####################### Fourier space and real space equal time correlation functions ##################################

	# Generate 2d points for radially averaged Fourier transform computations
	def makeQrad(self,dq,qmax,nq):
		nq2=int(2**0.5*nq)
		qmax2=2**0.5*qmax
		qx=np.linspace(0,qmax,nq)
		qy=np.linspace(0,qmax,nq)
		qrad=np.linspace(0,qmax2,nq2)
		# do this silly counting once and for all
		binval=np.empty((nq,nq))
		for kx in range(nq):
			for ky in range(nq):
				qval=np.sqrt(qx[kx]**2+qy[ky]**2)
				binval[kx,ky]=round(qval/dq)
		ptsx=[]
		ptsy=[]
		# do the indexing arrays
		for l in range(nq2):
			pts0x=[]
			pts0y=[]
			for kx in range(nq):
				hmm=np.nonzero(binval[kx,:]==l)[0]
				for v in range(len(hmm)):
					pts0y.append(hmm[v])
					pts0x.append(kx)
			ptsx.append(pts0x)
			ptsy.append(pts0y)
		return qx, qy, qrad, ptsx, ptsy
	
	# Static structure factor
	# Which is implicitly in 2D!!
	def FourierTrans(self,qmax=0.3,whichframe=1,usetype='all',L=self.geom.Lx,verbose=True):
		
		if not self.geom.manifold == 'plane':
			print("Configuration::FourierTrans - Error: attempting to compute 2d radially averaged Fourier transform on a non-flat surface. Stopping.")
			break
		
		# Note to self: only low q values will be interesting in any case. 
		# The stepping is in multiples of the inverse box size. Assuming a square box.
		print "Fourier transforming positions"
		dq=2*np.pi/L
		nq=int(qmax/dq)
		print "Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		#print " After Qrad" 
		fourierval=np.zeros((nq,nq),dtype=complex)
		
		if self.multiopt=="single":
			useparts = getUseparts(self,usetype)
			N = len(useparts)
			print(N)
			for kx in range(nq):
				for ky in range(nq):
					# And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
					fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.rval[useparts,0]+qy[ky]*self.rval[useparts,1])))/N
			
		else:
			useparts = getUseparts(self,usetype,whichframe)
			N = len(useparts)
			print(N)
			for kx in range(nq):
				for ky in range(nq):
					# And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
					fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.rval[whichframe,useparts,0]+qy[ky]*self.rval[whichframe,useparts,1])))/N
		plotval=N*(np.real(fourierval)**2+np.imag(fourierval)**2)
		
		#print plotval
		# Produce a radial averaging to see if anything interesting happens
		nq2=int(2**0.5*nq)
		valrad=np.zeros((nq2,))
		for l in range(nq2):
			valrad[l]=np.mean(plotval[ptsx[l],ptsy[l]])
		
		if verbose:
			plt.figure()
			plt.pcolor(qx,qy,plotval, vmin=0, vmax=3)
			plt.colorbar()
			plt.title('Positions')
			
			plt.figure()
			plt.plot(qrad,valrad)
			plt.xlabel('q')
			plt.ylabel('S(q)')
			plt.title('Positions')
			
		return qrad,valrad
	  
	
	def FourierTransVel(self,qmax=0.3,whichframe=1,usetype='all',L=self.geom.Lx,verbose=True):
		
		if not self.geom.manifold == 'plane':
			print("Configuration::FourierTransVel - Error: attempting to compute 2d radially averaged Fourier transform on a non-flat surface. Stopping.")
			break
		
		# Note to self: only low q values will be interesting in any case. 
		# The stepping is in multiples of the inverse box size. Assuming a square box.
		print "Fourier transforming velocities"
		# Note: Factor of 2 compared to static structure factor. Why?
		dq=np.pi/L
		nq=int(qmax/dq)
		print "Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		#print " After Qrad" 
		fourierval=np.zeros((nq,nq,2),dtype=complex)
		
		if self.multiopt=="single":
			useparts = getUseparts(self,usetype)
			N = len(useparts)
			print(N)
			for kx in range(nq):
				for ky in range(nq):
					fourierval[kx,ky,0]=np.sum(np.exp(1j*(qx[kx]*self.rval[useparts,0]+qy[ky]*self.rval[useparts,1]))*self.vval[useparts,0])/N
					fourierval[kx,ky,1]=np.sum(np.exp(1j*(qx[kx]*self.rval[useparts,0]+qy[ky]*self.rval[useparts,1]))*self.vval[useparts,1])/N 
		else:
			useparts = getUseparts(self,usetype,whichframe)
			N = len(useparts)
			print(N)
			for kx in range(nq):
				for ky in range(nq):
					fourierval[kx,ky,0]=np.sum(np.exp(1j*(qx[kx]*self.rval[whichframe,useparts,0]+qy[ky]*self.rval[whichframe,useparts,1]))*self.vval[whichframe,useparts,0])/N
					fourierval[kx,ky,1]=np.sum(np.exp(1j*(qx[kx]*self.rval[whichframe,useparts,0]+qy[ky]*self.rval[whichframe,useparts,1]))*self.vval[whichframe,useparts,1])/N 
			
		# Sq = \vec{v_q}.\vec{v_-q}, assuming real and symmetric
		# = \vec{v_q}.\vec{v_q*} = v
		Sq=np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2+np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2
		Sq=Nuse*Sq
		# Produce a radial averaging to see if anything interesting happens
		nq2=int(2**0.5*nq)
		Sqrad=np.zeros((nq2,))
		for l in range(nq2):
			Sqrad[l]=np.mean(Sq[ptsx[l],ptsy[l]])
		
		plotval_x=np.sqrt(np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2)
		plotval_y=np.sqrt(np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2)
		# Produce a radial averaging to see if anything interesting happens
		valrad=np.zeros((nq2,2))
		for l in range(nq2):
			valrad[l,0]=np.mean(plotval_x[ptsx[l],ptsy[l]])
			valrad[l,1]=np.mean(plotval_y[ptsx[l],ptsy[l]])

		print verbose
		print "Before plotting!"
		if verbose:
			#plt.figure()
			print "after first plotting command"
			#plt.pcolor(qx,qy,plotval_x)
			#print "after first plotting command"
			#plt.colorbar()
			#plt.title('Velocities - x')
			#plt.figure()
			#plt.pcolor(qx,qy,plotval_y)
			#plt.colorbar()
			#plt.title('Velocities - y')
		return qrad,valrad,Sqrad
	
	# Real space velocity correlation function
	# Note that this can work in higher dimensions. Uses geodesic distance, i.e. on the sphere if necessary
	def getVelcorrSingle(self,dx,xmax,whichframe=1,usetype='all',verbose=True):
		# start with the isotropic one - since there is no obvious polar region
		# and n is not the relevant variable, and v varies too much
		print "Velocity correlation function for frame " + str(whichframe)
		npts=int(round(xmax/dx))
		bins=np.linspace(0,xmax,npts)
		velcorr=np.zeros((npts,))
		velcount=np.zeros((npts,))
		
		if self.multiopt=="single":
			useparts = getUseparts(self,usetype)
			N = len(useparts)
			velav=np.sum(self.vval[useparts,:],axis=0)/N
			for k in range(N):
				vdot=np.sum(self.vval[useparts[k],:]*self.vval[useparts,:],axis=1)
				dr=self.geom.GeodesicDistance12(self.rval[useparts[k],:],self.rval[useparts,:])
				drbin=(np.round(dr/dx)).astype(int)
				for l in range(npts):
					pts=np.nonzero(drbin==l)[0]
					velcorr[l]+=sum(vdot[pts])
					velcount[l]+=len(pts)
		else:
			useparts = getUseparts(self,usetype,whichframe)
			N = len(useparts)
			velav=np.sum(self.vval[whichframe,useparts,:],axis=0)/N
			for k in range(N):
				vdot=np.sum(self.vval[whichframe,useparts[k],:]*self.vval[whichframe,useparts,:],axis=1)
				dr=self.geom.GeodesicDistance12(self.rval[whichframe,useparts[k],:],self.rval[whichframe,useparts,:])
				drbin=(np.round(dr/dx)).astype(int)
				for l in range(npts):
					pts=np.nonzero(drbin==l)[0]
					velcorr[l]+=sum(vdot[pts])
					velcount[l]+=len(pts)
				
		isdata=[index for index, value in enumerate(velcount) if value>0]
		velcorr[isdata]=velcorr[isdata]/velcount[isdata] - np.sum(velav*velav)
		if verbose:
			fig=plt.figure()
			isdata=[index for index, value in enumerate(velcount) if value>0]
			plt.plot(bins[isdata],velcorr[isdata],'.-')
			#plt.show()
			plt.xlabel("r-r'")
			plt.ylabel('Correlation')
		return bins,velcorr
	 
	 
	 
	## this should be in profiles
	#def getStatsBand(self,debug=False):
		#ez = np.array([0,0,1])  # lab frame z-axis
		## The order parameter with v_0 still in it. Normalize in final polish
		#orderparV=np.sum(vval,axis=0)/len(vval)
		#orderpar=np.sum(nval,axis=0)/len(nval)
		#print orderpar
		#print orderparV
		#direction = orderpar/np.linalg.norm(orderpar)
		#directionV = orderparV/np.linalg.norm(orderparV)
		#axisorth= np.cross(direction,directionV)
		#axisval=np.linalg.norm(axisorth)
		#alpha=np.arcsin(axisval)
		#axisorth=axisorth/axisval
		#axisnorm=np.cross(ez,directionV)
		#axisnorm/=np.linalg.norm(axisnorm)
		
		#print directionV
		#print axisorth
		
		#vel = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2 + self.vval[:,2]**2)
		#velnorm=((self.vval).transpose()/(vel).transpose()).transpose()
		
		#eng, press,stress = self.compute_energy_and_pressure()
		#print np.shape(stress)
		## Project the stresses into the e,theta,phi components. The rr component hast to be 0, and the r cross components
		## belong to the projection. So they are not all that interesting. 
		## We want the theta theta, theta phi, phi theta ant phi phi components (implicitly testing symmetries ...)
		## I give up on the notation. Stress is (N,3,3), the axes are (N,3). We want e_i sigma_ij e_j
		#s_tt=np.sum(axisnorm*np.einsum('kij,j->ki',stress,axisnorm),axis=1)
		#s_tp=np.sum(axisnorm*np.einsum('kij,j->ki',stress,directionV),axis=1)
		#s_pt=np.sum(directionV*np.einsum('kij,j->ki',stress,axisnorm),axis=1)
		#s_pp=np.sum(directionV*np.einsum('kij,j->ki',stress,directionV),axis=1)
		#print np.shape(s_tt)
		## Mean density really makes no sense? Determined by the initial conditions in periodic boundary conditions.
		## I do not wish to set up artificial bins in a translationally invariant system
		#vel_av=np.mean(vel)
		#eng_av=np.mean(eng)
		#press_av=np.mean(press)
		#s_tt_av=np.mean(s_tt)
		#s_tp_av=np.mean(s_tp)
		#s_pt_av=np.mean(s_pt)
		#s_pp_av=np.mean(s_pp)
		
		## Debugging output
		#if debug==True:
			#if HAS_MATPLOTLIB:
				#fig = plt.figure()
				#ax = fig.add_subplot(111, projection='3d')
				#ax.scatter(rval[:,0], rval[:,1], rval[:,2], zdir='z', c='b')
			#else:
				#print 'Error: Matplotlib does not exist on this machine, cannot plot system'
			
		#return [vel_av,eng_av,press_av,s_tt_av,s_tp_av,s_pt_av,s_pp_av,alpha,direction,directionV,orderpar,orderparV]
	
