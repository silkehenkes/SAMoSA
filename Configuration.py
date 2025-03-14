
from Geometry import *
from Interaction import *

#from read_param import *
from read_data_csv import *
from CellList import *

import sys,glob


try:
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	from matplotlib.pyplot import cm
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
		#print(' kwargs: ', kwargs)
		self.initype = kwargs["initype"]
		if self.initype == "fromCSV":
			self.fromCSV(kwargs["param"],kwargs["datapath"],kwargs["multiopt"])
		elif self.initype == "makeChild":
			self.makeChild(kwargs["parentconf"],kwargs["frame"],kwargs["usetype"],kwargs["makeCellList"],kwargs["redobox"])
		elif self.initype == "fromPython":
			self.fromPython(kwargs["param"],kwargs["rval"],kwargs["vval"],kwargs["nval"],kwargs["radii"],kwargs["ptype"],kwargs["flag"],kwargs["makeCellList"],kwargs["redobox"])
		else:
			print("Configuration:Unknown configuration constructor option, stopping!")
			sys.exit()
			
			
	# Use methods for the actual initialisation
	def fromCSV(self,param,datapath,multiopt,ignore=True,debug=False):
		self.debug = debug
		self.param = param
		self.datapath = datapath
		print(self.datapath)
		self.multiopt = multiopt
		self.ignore = True
		# Options for running mode the analysis
		if self.multiopt == "single":
			print("Configuring to read single files")
		elif self.multiopt == "many":
			print("Configuring to work on many files")
		else:
			print("Unknown file mode configuration option, stopping!")
			sys.exit()
		# Create the right geometry environment:
		self.geom=geometries[self.param.constraint](self.param)
		print(self.geom)
		
	# Create configuration from some python data passed down to here
	def fromPython(self,param,rval,vval,nval,radius,ptype,flag,makeCellList=True,redobox=False):
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
		
		# For defect tracking
		vnorm = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2+self.vval[:,2]**2)
		self.vhat = self.vval / np.outer(vnorm,np.ones((3,)))
		
		self.N = len(radius)
		self.sigma = np.mean(radius)
		print("New sigma is " + str(self.sigma))
		self.monodisperse = False
		
		
		# Generating new interaction assuming that it's not monodisperse and contains a single k
		#def __init__(self,param,sigma,ignore=False,debug=False):
		self.inter=Interaction(self.param,self.sigma,True,False)
		print (param.box)
		self.geom=geometries[param.constraint](param)
		
		if makeCellList:
			# Cut off the boxes. Curently only on  minimal z, for cornea. Others is less of a problem
			#if self.geom.manifold == 'sphere':
			#	zmin = np.amin(self.rval[:,2])-self.sigma
			#else:
			#	zmin = 'all'
			if redobox:
				rmin = np.zeros((3,))
				rmax = np.zeros((3,))
				for j in range(3):
					rmin[j] = np.amin(self.rval[:,j])-self.sigma
					rmax[j] = np.amax(self.rval[:,j])+self.sigma
				# def makeCellList(self,frame=1,rmin='default',rmax='default'):
				self.makeCellList(1,rmin,rmax)
			else:
				self.makeCellList(1)
		
		
				
	# Glorified copy constructor to isolate one frame and a subset of particles. Will serve as input to Hessian or Tesselation
	def makeChild(self,parentconf,frame=1,usetype='all',makeCellList = True, redobox=False):
		self.debug = parentconf.debug
		self.param = parentconf.param
		# Single frame for the child
		self.multiopt = "single"
		# And we have already narrowed down to the types we are looking for
		self.usetype = "all"
		self.sigma = parentconf.sigma
		self.monodisperse = parentconf.monodisperse
		
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
			
		# For defect tracking
		vnorm = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2+self.vval[:,2]**2)
		self.vhat = self.vval / np.outer(vnorm,np.ones((3,)))
		
		# Generating new interaction assuming that it's not monodisperse and contains a single k
		#def __init__(self,param,sigma,ignore=False,debug=False):
		self.inter=Interaction(self.param,self.sigma,False,True)
		#self.geom=geometries[param.constraint](param)
		#self.inter = parentconf.inter
		self.geom = parentconf.geom
		
		if makeCellList:
			if redobox:
				rmin = np.zeros((3,))
				rmax = np.zeros((3,))
				rmin[0] = np.amin(self.rval[:,0])-self.sigma
				rmin[1] = np.amin(self.rval[:,1])-self.sigma
				rmin[2] = np.amin(self.rval[:,2])-self.sigma
				rmax[0] = np.amax(self.rval[:,0])+self.sigma
				rmax[1] = np.amax(self.rval[:,1])+self.sigma
				rmax[2] = np.amax(self.rval[:,2])+self.sigma
				self.makeCellList(1,rmin,rmax)
				# def makeCellList(self,frame=1,rmin='default',rmax='default'):
				
			else:
				self.makeCellList(1)
		
		
	def readDataSingle(self,filename,dialect,internal=False,readtypes='all'):
		print("Processing file : " + filename)
		rd = ReadData(filename,dialect)
		
		# Using a pandas dataframe now
		x0 = rd.data["x"]
		x = rd.data["x"].to_numpy()
		y = rd.data["y"].to_numpy()
		# CCPy and others don't have z coordinate
		if "z" in rd.data.columns:
			z = rd.data["z"].to_numpy()
		else:
			z = np.zeros((len(x),))
			
		vx = rd.data["vx"].to_numpy()
		vy = rd.data["vy"].to_numpy()
		# CCPy and others don't have z coordinate
		if "vz" in rd.data.columns:
			vz = rd.data["vz"].to_numpy()
		else:
			vz = np.zeros((len(x),))
			
		nx = rd.data["nx"].to_numpy()
		ny = rd.data["ny"].to_numpy()
		# CCPy and others don't have z coordinate
		if "nz" in rd.data.columns:
			nz = rd.data["nz"].to_numpy()
		else:
			nz = np.zeros((len(x),))
			
		if "type" in rd.data.columns:
			ptype0 = rd.data["type"].to_numpy()
			if readtypes == 'all':
				N = len(x)
				ptype = ptype0
				useparts = range(N)
			else:
				useparts=[]
				for v in range(len(ptype0)):
					if ptype0[v] in readtypes:
						useparts.append(v)
				if len(useparts)==0:
					print('Error: No particles of the correct types in simulation, looking for ' + str(readtypes))
					sys.exit()
				else:
					N=len(useparts)
					hmm = np.array(ptype0)
					ptype = hmm[useparts]
		else:
			if readtypes == 'all':
				N = len(x)
				ptype = np.ones((N,))
				useparts = range(N)
			else:
				print("Error: looking for data with types " + str(readtypes) + " but data has no type information.")
				sys.exit()
		# cast to integer to make sure
		ptype = ptype.astype(int)
		
		rval = np.column_stack((x[useparts],y[useparts],z[useparts]))
		vval = np.column_stack((vx[useparts],vy[useparts],vz[useparts]))
		nval = np.column_stack((nx[useparts],ny[useparts],nz[useparts]))
		
		if not "radius" in rd.data.columns: 
			radius = np.ones(N)
			monodisperse=True
			sigma = 1.0
		else: 
			radius0 = rd.data["radius"].to_numpy()
			radius = radius0[useparts]
			sigma = np.mean(radius)
			monodisperse=False
		
		if "flag" in rd.data.columns:
			flag0 = rd.data["flag"].to_numpy()
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
			self.monodisperse = monodisperse
			self.ptype = ptype
			self.flag = flag
		
			# Create the Interaction class
			# def __init__(self,param,sigma,ignore=False,debug=False):
			self.inter=Interaction(self.param,self.sigma,self.ignore)
			
			# Apply periodic geomtry conditions just in case (there seem to be some rounding errors floating around)
			if self.geom.periodic:
				self.rval=self.geom.ApplyPeriodic2d(self.rval)
			
			if self.debug:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(self.rval[:,0], self.rval[:,1], self.rval[:,2], zdir='z', c='b')
		
	
	# read in data based on the content of the folder
	# readtypes is now a list (when it is not 'all'): Read data of these types, which contains the tracer options
	# tracers is type = [2], cornea is types = [1, 2] 
	def readDataMany(self,dialect,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all',filepattern='frame'):
		self.Nvariable=Nvariable
		#try:
		#	filepattern = self.param.dumpname
		#except:
		#	pass
		#	filepattern = 'frame'
		print(filepattern)
		files0 = sorted(glob.glob(self.datapath + filepattern + '*.dat'))
		if len(files0) == 0:
  			files0 = sorted(glob.glob(self.datapath + filepattern + '*.dat.gz'))
		if len(files0) == 0:
  			files0 = sorted(glob.glob(self.datapath + filepattern + '*.csv'))
		if len(files0) == 0:
			print("Error: could not identify data file name pattern, does not end in .dat or .dat.gz, or is not part of parameters, and is not 'frame'. Stopping.")
			sys.exit()
		
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
			self.Nval=np.zeros((self.Nsnap,),dtype='int')
			u=0
			for f in files:
				#print "Pre - Processing file : ", f
				rd = ReadData(f,dialect)
				x = rd.data["x"].to_numpy()
				if readtypes == 'all':
					self.Nval[u]=len(x)
				else:
					if "type" in rd.data.columns:
						ptype = rd.data["type"].to_numpy()
						useparts = []
						for v in range(len(ptype)):
							if ptype[v] in readtypes:
								useparts.append(v)
						if len(useparts)==0:
							print('Error: No particles of the correct types in simulation, looking for ' + str(readtypes))
							sys.exit()
						else:
							self.Nval[u]=len(useparts)
					else:
						print("Error: looking for data with types " + str(readtypes) + " but data has no type information.")
						sys.exit()
				u+=1
			self.N=int(np.amax(self.Nval))
		else:
			self.N,rval,vval,nval,radius,ptype,flag,self.sigma,monodisperse = self.readDataSingle(files[0],dialect,True,readtypes)
			self.Nval=[self.N for i in range(self.Nsnap)]
		print("Handling a total of maximum " + str(self.N) + " particles!")
			
		
		self.rval=np.zeros((self.Nsnap,self.N,3))
		self.vval=np.zeros((self.Nsnap,self.N,3))
		self.nval=np.zeros((self.Nsnap,self.N,3))
		self.flag=np.zeros((self.Nsnap,self.N))
		self.radius=np.zeros((self.Nsnap,self.N))
		self.ptype=np.zeros((self.Nsnap,self.N),dtype='int')
		u=0
		self.sigma = 0.0
		for f in files:
			# first read the data, for all types
			#return N,rval,vval,nval,radius,ptype,flag,sigma,monodisperse
			N,rval,vval,nval,radius,ptype,flag,sigma,monodisperse = self.readDataSingle(f,dialect,True,readtypes)
			# Running tab on maximum particle size for CellList
			if sigma>self.sigma:
				self.sigma = sigma
			# Then arrange in a sensible shape
			self.rval[u,:self.Nval[u],:]=rval
			self.vval[u,:self.Nval[u],:]=vval
			self.nval[u,:self.Nval[u],:]=nval
			self.flag[u,:self.Nval[u]]=flag
			self.radius[u,:self.Nval[u]]=radius
			self.ptype[u,:self.Nval[u]]=ptype
			u+=1
		self.monodisperse=monodisperse
		# Create the Interaction class
		#__init__(self,param,sigma,ignore=False,debug=False):
		self.inter=Interaction(self.param,self.sigma,self.ignore)
			
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
			for v in range(len(ptype)):
				if ptype[v] in usetype:
					useparts.append(v)
			if len(useparts)==0:
				print("Configuration::getUseparts - Warning: no particles of the desired type(s) " + str(usetype))
			return useparts
			
	# Just a printing helper function to see what my configuration is doing
	def printDiagnostic(self):
		print("Diagnostic of configuration: Set up to work on " + self.multiopt + " files.")
		if self.multiopt=="single":
			print("Working on " + str(self.N) + " particles of types " + str(np.unique(self.ptype)))
		else:
			print("Working with " + str(self.Nsnap) + " frames, with on average " + str(np.average(self.Nval)) + " particles of type " + str(np.unique(self.ptype)))
					
	# create Cell list, but only ever for a single frame (?)
	def makeCellList(self,frame=1,rmin='default',rmax='default'):
		# Delete any existing cell list
		try:
			del self.clist
		except:
			pass
		# Create the cell list, based on either neighbour list cutoff if defined, or else mean radius
		try:
			cellsize=param.nlist_rcut
		except:
			cellsize = 2*self.sigma
		if cellsize>5*self.sigma:
			cellsize=5*self.sigma
			print("Warning! Reduced the cell size to manageable proportions (5 times mean radius). Re-check if simulating very long objects!")
		# def __init__(self,geom, r_cut,rmin='default',rmax='default'):
		self.clist=CellList(self.geom,cellsize,rmin,rmax)
		# Populate it with all the particles:
		if self.multiopt=="single":
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
		if self.multiopt=="single":
			self.x1,self.x2,self.e1,self.e2=self.geom.TangentBundle(self.rval)
			return self.x1,self.x2,self.e1,self.e2
		else:
			self.x1,self.x2,self.e1,self.e2=self.geom.TangentBundle(self.rval[frame,self.Nval[frame],:])
			return self.x1,self.x2,self.e1,self.e2
	
	# Rotate the frame (for bands on spheres). If in multimodus, indicate frame as well.
	def rotateFrame(self,axis,rot_angle,frame=1,makeCellList=True):
		print ("Rotating frame with axis " + str(axis[0,:]) + " and angle " + str(rot_angle))
		if self.multiopt=="single":
			self.rval = self.geom.RotateRodriguez(self.rval,axis,rot_angle)
			self.vval = self.geom.RotateRodriguez(self.vval,axis,rot_angle)
			self.vhat = self.geom.RotateRodriguez(self.vhat,axis,rot_angle)
			self.vhat=((self.vhat).transpose()/(np.sqrt(np.sum(self.vhat**2,axis=1))).transpose()).transpose()
			self.nval = self.geom.RotateVectorial(self.nval,axis,rot_angle)
			self.nval=((self.nval).transpose()/(np.sqrt(np.sum(self.nval**2,axis=1))).transpose()).transpose()
			#self.vel = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2 + self.vval[:,2]**2)
			if makeCellList:
				# redo the cell list
				# This needs a mandatory new box now
				rmin = np.zeros((3,))
				rmax = np.zeros((3,))
				rmin[0] = np.amin(self.rval[:,0])-self.sigma
				rmin[1] = np.amin(self.rval[:,1])-self.sigma
				rmin[2] = np.amin(self.rval[:,2])-self.sigma
				rmax[0] = np.amax(self.rval[:,0])+self.sigma
				rmax[1] = np.amax(self.rval[:,1])+self.sigma
				rmax[2] = np.amax(self.rval[:,2])+self.sigma
				# def makeCellList(self,frame=1,rmin='default',rmax='default'):
				self.makeCellList(1,rmin,rmax)
		else:
			rval0 = self.geom.RotateRodriguez(self.rval[frame,self.Nval[frame],:],axis,rot_angle)
			vval0 = self.geom.RotateRodriguez(self.vval[frame,self.Nval[frame],:],axis,rot_angle)
			nval0 = self.geom.RotateRodriguez(self.nval[frame,self.Nval[frame],:],axis,rot_angle)
			nval0=((nval0).transpose()/(np.sqrt(np.sum(nval0**2,axis=1))).transpose()).transpose()
			self.rval[frame,self.Nval[frame],:]=rval0
			self.vval[frame,self.Nval[frame],:]=vval0
			self.nval[frame,self.Nval[frame],:]=nval0
			if makeCellList:
				# redo the cell list
				rmin = np.zeros((3,))
				rmax = np.zeros((3,))
				rmin[0] = np.amin(self.rval0[:,0])-self.sigma
				rmin[1] = np.amin(self.rval0[:,1])-self.sigma
				rmin[2] = np.amin(self.rval0[:,2])-self.sigma
				rmax[0] = np.amax(self.rval0[:,0])+self.sigma
				rmax[1] = np.amax(self.rval0[:,1])+self.sigma
				rmax[2] = np.amax(self.rval0[:,2])+self.sigma
				# def makeCellList(self,frame=1,rmin='default',rmax='default'):
				self.makeCellList(frame,rmin,rmax)

			
	# get neighbours of a particle
	# Careful! This needs a cell list
	# Issues with particle on top of each other, usually beacuse of sloppy implementation of glued boundary conditions
	# Remove them at the source, with a warning
	def getNeighbours(self,i,mult=1.0,dmax="default",frame=1,eps=1e-8):
		#print('particle:' + str(i))
		#print('mult:' + str(mult))
		#print('dmax:' + str(dmax))
		if dmax =="default":
			dmax = 2*self.sigma
		# Find potential neighbours from neighbour list first
		if self.multiopt=="single":
			cneighbours=self.clist.get_neighbours(self.rval[i,:])
			cneighbours.remove(i)
			#print "Cell list neighbours: " + str(len(cneighbours))
			drvec0=self.geom.ApplyPeriodic2d(self.rval[cneighbours,:]-self.rval[i,:])
		else:
			cneighbours=self.clist.get_neighbours(self.rval[frame,i,:])
			cneighbours.remove(i)
			#print "Cell list neighbours: " + str(len(cneighbours))
			drvec0=self.geom.ApplyPeriodic2d(self.rval[frame,cneighbours,:]-self.rval[frame,i,:])
		dist=np.sqrt(drvec0[:,0]**2+drvec0[:,1]**2+drvec0[:,2]**2)
		# Remove on top if there. Unpleasant slowdown. 
		ontop = [index for index, value in enumerate(dist) if value < eps]
		if len(ontop)>0:
			print('ontop:'+str(ontop))
			#print('cneighbours:'+str(cneighbours))
			for o in ontop:
				cneighbours.pop(o)
			#print('cneighbours:'+str(cneighbours))
			# remove and recompute
			if self.multiopt=="single":
				drvec0=self.geom.ApplyPeriodic2d(self.rval[cneighbours,:]-self.rval[i,:])
			else:
				drvec0=self.geom.ApplyPeriodic2d(self.rval[frame,cneighbours,:]-self.rval[frame,i,:])
			dist=np.sqrt(drvec0[:,0]**2+drvec0[:,1]**2+drvec0[:,2]**2)
			#print('dist:'+str(dist))
		#print('cneighbours:' + str(cneighbours))
		#print('drvec0'+ str(drvec0))
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
		#print neighbours
		if self.multiopt=="single":
			drvec=self.geom.ApplyPeriodic2d(self.rval[neighbours,:]-self.rval[i,:])
		else:
			drvec=self.geom.ApplyPeriodic2d(self.rval[frame,neighbours,:]-self.rval[frame,i,:])
		#dr=np.sqrt(drvec[:,0]**2+drvec[:,1]**2+drvec[:,2]**2)
		#print('neighbours:' + str(neighbours))
		#print('drvec:' + str(drvec))
		#print('radi:' + str(radi))
		#print('radj:' + str(radj))
		return neighbours, drvec, radi, radj
	      
	def compute_energy_and_pressure(self,frame=1):
		# Needs  mandatory cell list
		# Do not try this at home if you're using large parts of the cornea
		self.makeCellList(frame)
		if self.multiopt=="single":
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
	
	# Because surprisingly it's nowhere else!
	# only meaningful on a plane, but enable generically
	# returns 3-vectors, take magnitude for order parameter
	def getVicsek(self,frame=1):
		if self.multiopt=="single":
			dirvicsek = np.average(self.nval[:self.N,:],axis=0)
			vel2av = np.average(self.vval[:self.N,0]**2+self.vval[:self.N,1]**2+self.vval[:self.N,2]**2)
			velvicsek = np.average(self.vval,axis=0)/np.sqrt(vel2av)
		else:
			dirvicsek = np.average(self.nval[frame,:self.Nval[frame],:],axis=0)
			vel2av = np.average(self.vval[frame,:self.Nval[frame],0]**2+self.vval[frame,:self.Nval[frame],1]**2+self.vval[frame,:self.Nval[frame],2]**2)
			velvicsek = np.average(self.vval[frame,:self.Nval[frame],:],axis=0)/np.sqrt(vel2av)
		return dirvicsek, velvicsek

	# Chiral measure for oscillating systems
	# look for value of n x v/v. Along z. Adapt with unit normal dot product if ever we do a sphere (later)
	# only on non-pinned particles (or non-boundary, or whatever)
	# extract both average (global chirality) as well as sqrt of mean square (local chirality)
	def getChiral(self,usetype,frame=1):
		useparts=self.getUseparts(usetype,frame)
		if self.multiopt=='single':
			vmag=np.sqrt(self.vval[useparts,0]**2+self.vval[useparts,1]**2+self.vval[useparts,2]**2)
			singamma = np.average((self.nval[useparts,0]*self.vval[useparts,1]-self.nval[useparts,1]*self.vval[useparts,0])/vmag)
			abssingamma = np.sqrt(np.average((self.nval[useparts,0]*self.vval[useparts,1]-self.nval[useparts,1]*self.vval[useparts,0])**2/vmag**2))
		else:
			vmag=np.sqrt(self.vval[frame,useparts,0]**2+self.vval[frame,useparts,1]**2+self.vval[frame,useparts,2]**2)
			singamma = np.average((self.nval[frame,useparts,0]*self.vval[frame,useparts,1]-self.nval[frame,useparts,1]*self.vval[frame,useparts,0])/vmag)
			abssingamma = np.sqrt(np.average((self.nval[frame,useparts,0]*self.vval[frame,useparts,1]-self.nval[frame,useparts,1]*self.vval[frame,useparts,0])**2/vmag**2))
		return singamma,abssingamma
	
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
	# for square systems
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

	# Generate 2d points for radially averaged Fourier transform computations
	# more generally, for potentially rectangular systems
	def makeQradRect(self,qmax):
		dqx=np.pi/self.geom.Lx
		dqy=np.pi/self.geom.Ly
		nqx=int(qmax/dqx)
		nqy=int(qmax/dqy)
		# Pythagoras here: Note that in qspace, the space stays square at a given qmax
		nq2=int(np.sqrt(nqx**2+nqy**2))
		qmax2=2**0.5*qmax
		qx=np.linspace(0,qmax,nqx)
		qy=np.linspace(0,qmax,nqy)
		qrad=np.linspace(0,qmax2,nq2)
		dq=qrad[2]-qrad[1]
		# do this silly counting once and for all
		binval=np.empty((nqx,nqy))
		for kx in range(nqx):
			for ky in range(nqy):
				qval=np.sqrt(qx[kx]**2+qy[ky]**2)
				binval[kx,ky]=round(qval/dq)
		ptsx=[]
		ptsy=[]
		# do the indexing arrays
		for l in range(nq2):
			pts0x=[]
			pts0y=[]
			for kx in range(nqx):
				hmm=np.nonzero(binval[kx,:]==l)[0]
				for v in range(len(hmm)):
					pts0y.append(hmm[v])
					pts0x.append(kx)
			ptsx.append(pts0x)
			ptsy.append(pts0y)
		return qx, qy, qrad, ptsx, ptsy
	
	# Static structure factor
	# Which is implicitly in 2D!!
	def FourierTrans(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=False):
		
		#if L=="default":
		#	L = self.geom.Lx
		#if not self.geom.manifold == 'plane':
		#	print("Configuration::FourierTrans - Error: attempting to compute 2d radially averaged Fourier transform on a non-flat surface. Stopping.")
		#	sys.exit()
		
		# Note to self: only low q values will be interesting in any case. 
		# The stepping is in multiples of the inverse box size. Assuming a square box.
		#print("Fourier transforming positions")
		#dq=np.pi/L
		#nq=int(qmax/dq)
		#print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
		#qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		#nqx=nq
		#nqy=nq
		#nq2=int(2**0.5*nq)

		# More general framework for rectangular boxes
		print("Fourier transforming positions")
		qx, qy, qrad, ptsx, ptsy=self.makeQradRect(qmax)
		nqx=len(qx)
		nqy=len(qy)
		nq2=len(qrad)
		print("Stepping Fourier transform with step x" + str(qx[1]-qx[0])+ " and y " + str(qy[1]-qy[0]) + ", resulting in " + str(nq2)+ " steps.")


		fourierval=np.zeros((nqx,nqy),dtype=complex)
		if self.multiopt=="single":
			useparts = self.getUseparts(usetype)
			N = len(useparts)
			for kx in range(nqx):
				for ky in range(nqy):
					# And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
					fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.rval[useparts,0]+qy[ky]*self.rval[useparts,1])))/N
			
		else:
			useparts = self.getUseparts(usetype,whichframe)
			N = len(useparts)
			for kx in range(nqx):
				for ky in range(nqy):
					# And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
					fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.rval[whichframe,useparts,0]+qy[ky]*self.rval[whichframe,useparts,1])))/N
		plotval=N*(np.real(fourierval)**2+np.imag(fourierval)**2)
		
		#print plotval
		# Produce a radial averaging to see if anything interesting happens

		valrad=np.zeros((nq2,))
		for l in range(nq2):
			valrad[l]=np.mean(plotval[ptsx[l],ptsy[l]])
		
		if verbose:
			plt.figure()
			plt.pcolor(qx,qy,plotval, vmin=0, vmax=3)
			plt.colorbar()
			plt.title('Static structure factor (2d)')
			
			plt.figure()
			plt.plot(qrad,valrad,'.-r',lw=2)
			plt.xlabel('q')
			plt.ylabel('S(q)')
			plt.title('Static structure factor (radial)')
			
		return qrad,valrad
	  
	
	def FourierTransVel(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=False):
		
		#if L=="default":
		#	L = self.geom.Lx
		#if not self.geom.manifold == 'plane':
		#	print("Configuration::FourierTransVel - Error: attempting to compute 2d radially averaged Fourier transform on a non-flat surface. Stopping.")
		#	sys.exit()
		
		# Note to self: only low q values will be interesting in any case. 
		# The stepping is in multiples of the inverse box size. Assuming a square box.
		#print("Fourier transforming velocities")
		#dq=np.pi/L
		#nq=int(qmax/dq)
		#print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
		#qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		#nqx=nq
		#nqy=nq
		#nq2=int(2**0.5*nq)

		# More general framework for rectangular boxes
		print("Fourier transforming velocities")
		qx, qy, qrad, ptsx, ptsy=self.makeQradRect(qmax)
		nqx=len(qx)
		nqy=len(qy)
		nq2=len(qrad)
		print("Stepping Fourier transform with step x" + str(qx[1]-qx[0])+ " and y " + str(qy[1]-qy[0]) + ", resulting in " + str(nq2)+ " steps.")

		fourierval=np.zeros((nqx,nqy,2),dtype=complex)
		
		if self.multiopt=="single":
			useparts = self.getUseparts(usetype)
			N = len(useparts)
			for kx in range(nqx):
				for ky in range(nqy):
					fourierval[kx,ky,0]=np.sum(np.exp(1j*(qx[kx]*self.rval[useparts,0]+qy[ky]*self.rval[useparts,1]))*self.vval[useparts,0])/N
					fourierval[kx,ky,1]=np.sum(np.exp(1j*(qx[kx]*self.rval[useparts,0]+qy[ky]*self.rval[useparts,1]))*self.vval[useparts,1])/N 
		else:
			useparts = self.getUseparts(usetype,whichframe)
			N = len(useparts)
			print(N)
			for kx in range(nqx):
				for ky in range(nqy):
					fourierval[kx,ky,0]=np.sum(np.exp(1j*(qx[kx]*self.rval[whichframe,useparts,0]+qy[ky]*self.rval[whichframe,useparts,1]))*self.vval[whichframe,useparts,0])/N
					fourierval[kx,ky,1]=np.sum(np.exp(1j*(qx[kx]*self.rval[whichframe,useparts,0]+qy[ky]*self.rval[whichframe,useparts,1]))*self.vval[whichframe,useparts,1])/N 
			
		# Sq = \vec{v_q}.\vec{v_-q}, assuming real and symmetric
		# = \vec{v_q}.\vec{v_q*} = v
		Sq=np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2+np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2
		#Sq=N*Sq
		# Produce a radial averaging to see if anything interesting happens
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
		if verbose:
			plt.figure()
			plt.plot(qrad,Sqrad,'.-r',lw=2)
			plt.xlabel('q')
			plt.ylabel('correlation')
			plt.title('Fourier space velocity correlation')
		return qrad,valrad,Sqrad
	
	# Real space velocity correlation function
	# Note that this can work in higher dimensions. Uses geodesic distance, i.e. on the sphere if necessary
	def getVelcorrSingle(self,dx,xmax,whichframe=1,usetype='all',verbose=True):
		# start with the isotropic one - since there is no obvious polar region
		# and n is not the relevant variable, and v varies too much
		print("Velocity correlation function for frame " + str(whichframe))
		npts=int(round(xmax/dx))
		bins=np.linspace(0,xmax,npts)
		velcorr=np.zeros((npts,))
		velcount=np.zeros((npts,))
		
		if self.multiopt=="single":
			useparts = self.getUseparts(usetype)
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
			useparts = self.getUseparts(usetype,whichframe)
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
			plt.plot(bins[isdata],velcorr[isdata],'.-r',lw=2)
			#plt.show()
			plt.xlabel("r-r'")
			plt.ylabel('Correlation')
			plt.title('Spatial velocity correlation')
		return bins,velcorr

	
