
from Configuration import *
from Hessian import *

# Dynamics is a child class of configuration
# All its methods are dynamical, i.e. they include some time-correlated information	
class Dynamics(Configuration):
	
	# Use the Configuration default constructor. We *need* to be set to self.multiopt == "many"!
	def validate_initialise(self):
		if not self.multiopt == "many":
			print("Dynamics:: Error: Cannot use dynamics without reading in multiple files! Aborting.")
			sys.exit()
		# Some booleans to avoid redoing the whole shebang twice as some computations depend on others
		self.hasMSD = False
		# Have we isolated the particles we are following? Implicit assumption: we are only ever tracking the 
		# same types of particles in a given instance.
		# If a problem, can always set it to 'False' again from outside
		self.tracked = False
		self.canDrift = False
		self.hasAverage = False
	
	# By design, this is only meaningful on the whole data set, e.g. do not subtract for tracer particles only
	def takeDrift(self):
		self.drift=np.zeros((self.Nsnap,3))
		if self.Nvariable:
			print("Dynamics:: Variable N: Taking off the drift is meaningless. Doing nothing.")
		else:	
			for u in range(1,self.Nsnap):
				dr=self.geom.ApplyPeriodic2d(self.rval[u,:,:]-self.rval[u-1,:,:])
				drift0=np.sum(dr,axis=0)/self.N
				self.drift[u,:]=self.drift[u-1,:]+drift0
				#print self.drift[u,:]
			self.canDrift = True
		# so as to stop trying to do stuff	

	# just compute averages once and for all 
	def makeAverage(self):
		self.raverage  = np.mean(self.rval[:,:,:],axis=0)
		self.hasAverage = True

	# Compute average positions and create a sub-configuration to feed to the Hessian
	def makeAverageChild(self,usetype):
		if self.Nvariable:
			print("Dynamics:: Variable N: Taking average particle positions is unsound. Stopping.")
			sys.exit()
		# If that's out of the way, nothing is being relabeled. These are our labels once and for all
		useparts = self.getUseparts(usetype,0)
		# This is the crucial one for the Hessian
		rval0 = np.mean(self.rval[:,useparts,:],axis=0)
		# Less meaningful, but indulge me
		vval0 = np.mean(self.vval[:,useparts,:],axis=0)
		nval0 = np.mean(self.nval[:,useparts,:],axis=0)
		# This part is not changing, hence use initial
		radius0 = self.radius[0,useparts]
		ptype0 = self.ptype[0,useparts]
		flag0 = self.flag[0,useparts]
		
		#self.fromPython(kwargs["param"],kwargs["rval"],kwargs["vval"],kwargs["nval"],kwargs["radii"],kwargs["ptype"],kwargs["flag"])
		#def fromPython(self,param,rval,vval,nval,radius,ptype,flag):
		averageChild = Configuration(initype="fromPython",param=self.param,rval=rval0,vval=vval0,nval=nval0,radii=radius0,ptype=ptype0,flag=flag0)
		return averageChild
		
	# Tracking a subset of particles: This identifies the relevant ones
	# and gives an error if that number changes or the flags don't match
	# complicated: if the particles we are aiming to track move around in their list index
	# Conceptual debug January 4 2023: Needs 'complicated' even if it shifts around as a block (without switching positions in reduced list)
	# Which is standard behaviour if it was printed at the bottom of the input file.
	def getTrack(self,usetype):
		if usetype == 'all':
			if self.Nvariable:
				print("Dynamics::getTrack - Attempting to track ALL particles while N is variable: this doesn't make sense. Stopping")
				sys.exit()
			else:
				self.Ntrack = self.N
				self.usethese = range(self.Ntrack)
				self.complicated = False
		else:
			useparts = self.getUseparts(usetype,0)
			flag0 = self.flag[0,useparts]
			self.Ntrack = len(useparts)
			usethese0 = np.empty((self.Nsnap,self.Ntrack),dtype='int')
			# If the positions of our flagged particles are jumping around, we are in deep doodoo for vectorisation
			self.complicated = False
			for u in range(0,self.Nsnap):
				#print("Looking at snapshot" + str(u))
				useparts0 = useparts 
				useparts = self.getUseparts(usetype,u)
				N = len(useparts)
				if N != self.Ntrack:
					print("Dynamics::getTrack - Error: Number of tracked particles is changing, check that chosen type is not dividing or dying!")
					sys.exit()
				# preliminarily, stick array as is into usethese0. Revise if relabeled.
				usethese0[u,:] = useparts
				# Check that 1. These are the same labels at the same places. Can continue to use simple vectorisation in that case
				# Else see if it's a shifted block or a rearrangement
				if not np.array_equal(useparts,useparts0):
					self.complicated=True
				flag = self.flag[u,useparts]
				if not np.array_equal(flag,flag0):
					# it's already complicated like this!
					#self.complicated = True
					# However, they could still be there, but relabeled. Then things are non-vectorisable, but possible, though slow
					if set(flag) == set(flag0):
						# now we need to get useparts in the same order as the initial flags
						# Better slow than wrong ...
						for k in flag:
							idx = flag0.index(k)
							usethese0[u,idx] = useparts[k]
					else:
						print("Dynamics::getTrack - Error: Labels (flags) of tracked particles are changing, something is wrong!")
						sys.exit()
			if self.complicated:
				self.usethese = usethese0
			else:
				self.usethese = useparts
		self.tracked=True
				
				
	# relative velocity distribution (and average velocity)
	# component wise as well, assumes x and y directions only
	def getVelDist(self,bins,bins2,usetype='all',verbose=True):
		vav=np.zeros((self.Nsnap,))
		vdist=np.zeros((len(bins)-1,))
		vdist2=np.zeros((len(bins2)-1,))
		for u in range(self.Nsnap):
			#print u
			# The particles we want to average over
			useparts = self.getUseparts(usetype,u)
			# get all the magnitudes, and all the components
			vmagnitude=np.sqrt(self.vval[u,useparts,0]**2+self.vval[u,useparts,1]**2+self.vval[u,useparts,2]**2)
			vcomponents = self.vval[u,useparts,0:2].flatten()
			# mean velocity magnitude
			vav[u]=np.mean(vmagnitude)
			# normalised magnitude histogram
			vdist0,dummy=np.histogram(vmagnitude/vav[u],bins,density=True)
			vdist+=vdist0
			# normalised component histogram
			vdist20,dummy=np.histogram(vcomponents/vav[u],bins2,density=True)
			vdist2+=vdist20
		vdist/=self.Nsnap
		vdist2/=self.Nsnap
		if verbose:
			fig=plt.figure()
			db=bins[1]-bins[0]
			plt.semilogy(bins[1:]-db/2,vdist,'r.-',lw=2)
			plt.xlabel('v/<v>')
			plt.ylabel('P(v/<v>)')
			plt.title('Scaled velocity magnitude distribution')
			
			fig=plt.figure()
			db=bins2[1]-bins2[0]
			plt.semilogy(bins2[1:]-db/2,vdist2,'r.-',lw=2)
			plt.xlabel('v/<v>')
			plt.ylabel('P(v/<v>)')
			plt.title('Scaled velocity component distribution')
			
			xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
			plt.figure()
			plt.plot(xval,vav,'r.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('mean velocity')
		return vav, vdist,vdist2		
	
		
####################################### Glassy Physics ###############################################################
	def getMSD(self,takeDrift,usetype='all',verbose=True):
		self.msd=np.empty((self.Nsnap,))
		
		# Careful, we need a sanity check: We can only compute a MSD if the particles we're tracking isn't changing (whatever the rest of the system is doing)
		if not self.tracked:
			self.getTrack(usetype)				
		
		if takeDrift:
			self.takeDrift()
			
		for u in range(self.Nsnap):	
			smax=self.Nsnap-u
			# Note that by design, take drift needs to work on the whole data set
			# so take drift only if 1) we want to 2) it's possible to calculate it
			if takeDrift and self.canDrift:
				if self.complicated:
					print("Dynamics::getMSD - Taking off drift at constant N while complicated labels: My head is hurting, stopping here. Reconsider what you're doing ... it's probably wrong")
					sys.exit()
				else:
					hmm=(self.drift[:smax,:]-self.drift[u:,:])
					takeoff=np.einsum('j,ik->ijk',np.ones((self.Ntrack,)),hmm)
				if self.geom.periodic:
					dr=self.geom.ApplyPeriodic3d(self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:])-takeoff[:,:,:]
				else:
					dr=self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:]-takeoff[:,:,:]
			else:
				if not self.complicated:
					if self.geom.periodic:
						dr=self.geom.ApplyPeriodic3d(self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:])
					else:
						dr=self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:]
				else:
					print("Dynamics::getMSD - Warning: Complicated relabeling: This is slow, make sure to only track a few particles")
					dr = np.zeros((smax,self.Ntrack,3))
					for s in range(smax):
						if self.geom.periodic:
							dr[s,:,:] = self.geom.ApplyPeriodic2d(self.rval[s,list(self.usethese[s,:]),:]-self.rval[u+s,list(self.usethese[u+s,:]),:])
						else:
							dr[s,:,:] = self.rval[s,list(self.usethese[s,:]),:]-self.rval[u+s,list(self.usethese[u+s,:]),:]
			self.msd[u]=np.sum(np.sum(np.sum(dr**2,axis=2),axis=1),axis=0)/(self.Ntrack*smax)
		self.hasMSD = True
					
		xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			fig=plt.figure()
			plt.loglog(xval,self.msd,'r.-',lw=2)
			plt.loglog(xval,self.msd[1]/(1.0*xval[1])*xval,'-',lw=2,color=[0.5,0.5,0.5])
			plt.xlabel('time')
			plt.ylabel('MSD')
			plt.title('Mean square displacement')
			#plt.show()
		
		return xval, self.msd
            
	# Velocity autocorrelation function
	def getVelAuto(self,usetype='all',verbose=False):
		self.velauto=np.empty((self.Nsnap,))
		self.v2av=np.empty((self.Nsnap,))
		
		if not self.tracked:
			self.getTrack(usetype)
			
		# First compute normalised velocities. Note: normalised by mean velocity in the whole system at that time, not unit vectors!
		vnormed = np.zeros((self.Nsnap,self.Ntrack,3))
		for u in range(self.Nsnap):
			if self.complicated:
				self.v2av[u]=np.sum(np.sum((self.vval[u,list(self.usethese[u,:]),:])**2,axis=1),axis=0)/(self.Ntrack)
				vnormed[u,:,:]=self.vval[u,list(self.usethese[u,:]),:]/np.sqrt(self.v2av[u])
			else:
				self.v2av[u]=np.sum(np.sum((self.vval[u,self.usethese,:])**2,axis=1),axis=0)/(self.Ntrack)
				vnormed[u,:,:]=self.vval[u,self.usethese,:]/np.sqrt(self.v2av[u])
				
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			if self.complicated:
				print("Dynamics::getVelAuto - Warning: Complicated relabeling: This is slow, make sure to only track a few particles") 
				for s in range(smax):
					self.velauto[u] += np.sum(vnormed[s,:,0]*vnormed[u+s,:,0]+vnormed[s,:,1]*vnormed[u+s,:,1]+vnormed[s,:,2]*vnormed[u+s,:,2])/(self.Ntrack*smax)
			else:
				self.velauto[u]=np.sum(np.sum((vnormed[:smax,:,0]*vnormed[u:,:,0]+vnormed[:smax,:,1]*vnormed[u:,:,1]+vnormed[:smax,:,2]*vnormed[u:,:,2]),axis=1),axis=0)/(self.Ntrack*smax)

                                
		xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			fig=plt.figure()
			plt.plot(xval,self.velauto,'r.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('correlation')
			plt.title('Normalised Velocity autocorrelation function')
			#plt.show()
		return xval, self.velauto, self.v2av

	# Director autocorrelation function (oscillating systems)
	def getDirAuto(self,usetype='all',verbose=False):
		# n dot n cos of angle
		self.dirauto=np.empty((self.Nsnap,))
		# while we are here, ocmpute chiral measure omega (rotation speed)
		# estimated from snapshots: theta = np.artcan2(ny,nx), omega = dtheta/dt, modulo 2pi
		self.omega = np.empty((self.Nsnap))
		if not self.tracked:
			self.getTrack(usetype)
			
		# Extract what we want just for ease of writing
		nnuse = np.zeros((self.Nsnap,self.Ntrack,3))
		for u in range(self.Nsnap):
			if self.complicated:
				nnuse[u,:,:]=self.nval[u,list(self.usethese[u,:]),:]
			else:
				nnuse[u,:,:]=self.nval[u,self.usethese,:]
				
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			if self.complicated:
				print("Dynamics::getVelAuto - Warning: Complicated relabeling: This is slow, make sure to only track a few particles") 
				for s in range(smax):
					self.dirauto[u] += np.sum(nnuse[s,:,0]*nnuse[u+s,:,0]+nnuse[s,:,1]*nnuse[u+s,:,1]+nnuse[s,:,2]*nnuse[u+s,:,2])/(self.Ntrack*smax)
			else:
				self.dirauto[u]=np.sum(np.sum((nnuse[:smax,:,0]*nnuse[u:,:,0]+nnuse[:smax,:,1]*nnuse[u:,:,1]+nnuse[:smax,:,2]*nnuse[u:,:,2]),axis=1),axis=0)/(self.Ntrack*smax)
		# chiral rotation rate
		omega = np.zeros((self.Nsnap-1))
		absomega = np.zeros((self.Nsnap-1))
		for u in range(1,self.Nsnap):
			theta = np.arctan2(nnuse[u,:,1],nnuse[u,:,0])
			theta_m1 = np.arctan2(nnuse[u-1,:,1],nnuse[u-1,:,0])
			dtheta = theta-theta_m1
			isperiodic = np.where(np.abs(dtheta)>np.pi)[0]
			dtheta[isperiodic] -=2*np.pi*np.sign(dtheta[isperiodic])
			omega[u-1] = np.average(dtheta)/self.param.dt
			absomega[u-1] = np.sqrt(np.average(dtheta**2))/self.param.dt
                
		xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			fig=plt.figure()
			plt.plot(xval,self.dirauto,'r.-',lw=2)
			plt.plot(xval[1:],omega,'k.-',lw=2)
			plt.plot(xval[1:],absomega,'g.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('correlation')
			plt.title('Normalised Director autocorrelation function')
			#plt.show()
		return xval, self.dirauto,omega,absomega

	# And for good measuer, diplacement correlators (implicitly assumes that we are glassy and that temporal mean = potential bottom)
	def getDispAuto(self,usetype='all',verbose=False):
		self.dispauto=np.empty((self.Nsnap,))
		self.disp2av=np.empty((self.Nsnap,))
		
		if not self.tracked:
			self.getTrack(usetype)
		if not self.hasAverage:
			self.makeAverage()
			
		# First compute normalised velocities. Note: normalised by mean velocity in the whole system at that time, not unit vectors!
		dispnormed = np.zeros((self.Nsnap,self.Ntrack,3))
		for u in range(self.Nsnap):
			
			if self.complicated:
				if self.geom.periodic:
					dr=self.geom.ApplyPeriodic2d(self.rval[u,list(self.usethese[u,:]),:]-self.raverage[list(self.usethese[u,:]),:])
				else:
					dr=self.rval[u,list(self.usethese[u,:]),:]-self.raverage[list(self.usethese[u,:]),:]
			else:
				if self.geom.periodic:
					dr=self.geom.ApplyPeriodic2d(self.rval[u,self.usethese,:]-self.raverage[self.usethese,:])
				else:
					dr=self.rval[u,self.usethese,:]-self.raverage[self.usethese,:]
			self.disp2av[u]=np.sum(np.sum((dr)**2,axis=1),axis=0)/(self.Ntrack)
			dispnormed[u,:,:]=dr/np.sqrt(self.disp2av[u])
				
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			if self.complicated:
				print("Dynamics::getVelAuto - Warning: Complicated relabeling: This is slow, make sure to only track a few particles") 
				for s in range(smax):
					self.dispauto[u] += np.sum(dispnormed[s,:,0]*dispnormed[u+s,:,0]+dispnormed[s,:,1]*dispnormed[u+s,:,1]+dispnormed[s,:,2]*dispnormed[u+s,:,2])/(self.Ntrack*smax)
			else:
				self.dispauto[u]=np.sum(np.sum((dispnormed[:smax,:,0]*dispnormed[u:,:,0]+dispnormed[:smax,:,1]*dispnormed[u:,:,1]+dispnormed[:smax,:,2]*dispnormed[u:,:,2]),axis=1),axis=0)/(self.Ntrack*smax)   

		xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			fig=plt.figure()
			plt.plot(xval,self.dispauto,'r.-',lw=2)
			plt.plot(xval,self.disp2av,'k.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('correlation')
			plt.title('Normalised Displacement autocorrelation function')
			#plt.show()
		return xval, self.dispauto, self.disp2av
        
	# Computes non-gaussian factor / kurtosis (4th moment), and also the MSD (2nd moment) if it hasn't been done yet
	# returns all three
	def getNonGaussian(self,takeDrift,usetype='all',verbose=True):
		
		kurtosis=np.empty((self.Nsnap,))
		nongaussian=np.empty((self.Nsnap,))
		
		if not self.tracked:
			self.getTrack(usetype)
		
		# Assume this has already happened for MSD, as the calculation needs to be done beforehand
		# Hidden assumption: we are doing MSD and kurtosis on the same particles (anything else is daft)
		if not self.hasMSD:
			self.msd=np.empty((self.Nsnap,))
			if takeDrift:
				self.takeDrift()
			
		for u in range(self.Nsnap):	
			smax=self.Nsnap-u
			# Note that by design, take drift needs to work on the whole data set
			if takeDrift and self.canDrift:
				if self.complicated:
					print("Dynamics::getNonGaussian - Taking off drift at constant N while complicated labels: My head is hurting, stopping here. Reconsider what you're doing ... it's probably wrong")
					sys.exit()
				else:
					hmm=(self.drift[:smax,:]-self.drift[u:,:])
					takeoff=np.einsum('j,ik->ijk',np.ones((self.Ntrack,)),hmm)
				if self.geom.periodic:
					dr=self.geom.ApplyPeriodic3d(self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:])-takeoff[self.usethese,:]
				else:
					dr=self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:]-takeoff[self.usethese,:]
			else:
				if not complicated:
					if self.geom.periodic:
						dr=self.geom.ApplyPeriodic3d(self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:])
					else:
						dr=self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:]
				else:
					print("Dynamics::getNonGaussian - Warning: Complicated relabeling: This is slow, make sure to only track a few particles")
					dr = np.zeros((smax,self.Ntrack,3))
					for s in range(smax):
						if self.geom.periodic:
							dr[s,:,:] = self.geom.ApplyPeriodic2d(self.rval[s,list(self.usethese[s,:]),:]-self.rval[u+s,list(self.usethese[u+s,:]),:])
						else:
							dr[s,:,:] = self.rval[s,list(self.usethese[s,:]),:]-self.rval[u+s,list(self.usethese[u+s,:]),:]
			if not self.hasMSD:
				self.msd[u]=np.sum(np.sum(np.sum(dr**2,axis=2),axis=1),axis=0)/(self.Ntrack*smax)
			kurtosis[u]=np.sum(np.sum(np.sum(dr**4,axis=2),axis=1),axis=0)/(self.Ntrack*smax)
                 
		# Check for dimensional scaling, prefactor in 2d == 2/3 [?]
		nongaussian=3.0/5.0*kurtosis/self.msd**2 -1
		xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			fig=plt.figure()
			plt.loglog(xval,self.msd,'r.-',lw=2)
			plt.loglog(xval,self.msd[1]/(1.0*xval[1])*xval,'-',lw=2,color=[0.5,0.5,0.5])
			plt.xlabel('time')
			plt.ylabel('MSD')
			
			fig=plt.figure()
			plt.loglog(xval,kurtosis,'r.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('kurtosis')
			
			fig=plt.figure()
			plt.loglog(xval,nongaussian,'r.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('Non - Gaussian')
			
			#plt.show()
		return xval, self.msd, kurtosis, nongaussian

	
	# Definition of the self-intermediate scattering function (Flenner + Szamel)
	# 1/N <\sum_n exp(iq[r_n(t)-r_n(0)]>_t,n
	def SelfIntermediate(self,qval,takeDrift,usetype='all',verbose=False):
		# This is single particle, single q, shifted time step. Equivalent to the MSD, really
		SelfInt=np.empty((self.Nsnap,),dtype=complex)
		
		if not self.tracked:
			self.getTrack(usetype)
			
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			if takeDrift and self.canDrift:
				if self.complicated:
					print("Dynamics::SelfIntermediate - Taking off drift at constant N while complicated labels: My head is hurting, stopping here. Reconsider what you're doing ... it's probably wrong")
					sys.exit()
				else:
					hmm=(self.drift[:smax,:]-self.drift[u:,:])
					takeoff=np.einsum('j,ik->ijk',np.ones((self.Ntrack,)),hmm)
				if self.geom.periodic:
					SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(self.geom.ApplyPeriodicX(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0])+takeoff[:,:,0])+1.0j*qval[1]*(self.geom.ApplyPeriodicY(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1])+takeoff[:,:,1])+1.0j*qval[2]*(self.geom.ApplyPeriodicZ(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2])+takeoff[:,:,2])),axis=1),axis=0)/(self.Ntrack*smax)
				else:
					SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0]+takeoff[:,:,0])+1.0j*qval[1]*(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1]+takeoff[:,:,1])+1.0j*qval[2]*(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2]+takeoff[:,:,2])),axis=1),axis=0)/(self.Ntrack*smax)
			else:
				if self.complicated:
					for s in range(smax):
						ps = list(self.usethese[s,:])
						pu = list(self.usethese[u+s,:])
						if self.geom.periodic:
							SelfInt[u] += np.sum(np.exp(1.0j*qval[0]*(self.geom.ApplyPeriodicX(-self.rval[s,ps,0]+self.rval[u+s,pu,0]))+1.0j*qval[1]*(self.geom.ApplyPeriodicY(-self.rval[s,ps,1]+self.rval[u+s,pu,1]))+1.0j*qval[2]*(self.geom.ApplyPeriodicZ(-self.rval[s,ps,2]+self.rval[u+s,pu,2]))))/(self.N*smax)
						else:
							SelfInt[u] += np.sum(np.sum(np.exp(1.0j*qval[0]*(-self.rval[s,ps,0]+self.rval[u+2,pu,0])+1.0j*qval[1]*(-self.rval[s,ps,1]+self.rval[u+s,pu,1])+1.0j*qval[2]*(-self.rval[s,ps,2]+self.rval[u+s,pu,2]))))/(self.Ntrack*smax)
				else:
					if self.geom.periodic:
						SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(self.geom.ApplyPeriodicX(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0]))+1.0j*qval[1]*(self.geom.ApplyPeriodicY(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1]))+1.0j*qval[2]*(self.geom.ApplyPeriodicZ(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2]))),axis=1),axis=0)/(self.N*smax)
					else:
						SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0])+1.0j*qval[1]*(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1])+1.0j*qval[2]*(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2])),axis=1),axis=0)/(Ntrack*smax)
                        
		
		#print tval
		# Looking at the absolute value of it here
		SelfInt2=(np.real(SelfInt)**2 + np.imag(SelfInt)**2)**0.5
		
		tval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			qnorm=np.sqrt(qval[0]**2+qval[1]**2+qval[2]**2)
			fig=plt.figure()
			plt.semilogx(tval,SelfInt2,'.-r',lw=2)
			plt.xlabel('time')
			plt.ylabel('F_s(k,t)')
			plt.title('Self-intermediate, k = ' + str(qnorm))
			#plt.show()
		return tval, SelfInt2

	def getDynStruct(self,qmax,omegamax,kind,usetype='all',L="default",verbose=True):
		if L=="default":
			L = self.geom.Lx
		if not self.geom.manifold == 'plane':
			print("Configuration::FourierTransVel - Error: attempting to compute 2d radially averaged Fourier transform on a non-flat surface. Stopping.")
			sys.exit()
		
		# Note to self: only low q values will be interesting in any case. 
		# The stepping is in multiples of the inverse box size. Assuming a square box.
		print("Fourier transforming velocities")
		dq=2*np.pi/L
		nq=int(qmax/dq)
		print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		dom=2*np.pi/(self.Nsnap*self.param.dt*self.param.dump['freq'])
		nom=int(omegamax/dom)
		print("Stepping time Fourier transform with step" + str(dom) + ",resulting in " + str(nom) + " steps.")
		omega=np.linspace(0,omegamax,nom)
		print(omega)

		#print " After Qrad" 
		fourierval=np.zeros((nq,nq,nom,2),dtype=complex)
		
		if not self.tracked:
			self.getTrack(usetype)
		if self.complicated:
			print("Attempting nasty spatiotemporal correlations which tracking complicated particles. This is too ineffective. stopping.")
			sys.exit()
		if kind=='displacement':
			if self.geom.periodic:
				field=self.geom.ApplyPeriodic2d(self.rval[:,self.usethese,:]-self.raverage[self.usethese,:])/np.sqrt(np.average(self.disp2av))
			else:
				field=self.rval[:,self.usethese,:]-self.raverage[self.usethese,:]/np.sqrt(np.average(self.disp2av))
			print('Maximum displacement: ' +np.str(np.amax(np.abs(field))))
			print('Minimum displacement: ' +np.str(np.amin(np.abs(field))))
		elif kind=='velocity':
			field=self.vval[:,self.usethese,:]/np.sqrt(np.average(self.v2av))
		elif kind=='director':
			field=self.nval[:,self.usethese,:]
		else:
			print("Unknown field in structure factor calculation. Stopping.")
			sys.exit()

		# need a time matrix with the right dimensions
		tmat=np.zeros((self.Nsnap,len(self.usethese)))
		for u in range(self.Nsnap):
			tmat[u,:]=u*self.param.dt*self.param.dump['freq']
		N = len(self.usethese)
		for om in range(nom):
			print (om,omega[om])
			for kx in range(nq):
				for ky in range(nq):
					fourierval[kx,ky,om,0]=np.sum(np.sum(np.exp(1j*(qx[kx]*self.rval[:,self.usethese,0]+qy[ky]*self.rval[:,self.usethese,1]+omega[om]*tmat))*field[:,:,0],axis=1))/(N*self.Nsnap)
					fourierval[kx,ky,om,1]=np.sum(np.sum(np.exp(1j*(qx[kx]*self.rval[:,self.usethese,0]+qy[ky]*self.rval[:,self.usethese,1]+omega[om]*tmat))*field[:,:,1],axis=1))/(N*self.Nsnap) 
			
		# Sq = \vec{v_q}.\vec{v_-q}, assuming real and symmetric
		# = \vec{v_q}.\vec{v_q*} = v
		Sq=np.real(fourierval[:,:,:,0])**2+np.imag(fourierval[:,:,:,0])**2+np.real(fourierval[:,:,:,1])**2+np.imag(fourierval[:,:,:,1])**2
		Sq=N*Sq
		# Produce a radial averaging to see if anything interesting happens
		nq2=int(2**0.5*nq)
		Sqrad=np.zeros((nq2,nom))
		for om in range(nom):
			for l in range(nq2):
				Sqrad[l,om]=np.average(Sq[ptsx[l],ptsy[l],om])
		
		if verbose:
			plt.figure()
			# because pcolor is stupid
			qplot=np.zeros((len(qrad)+1))
			omegaplot=np.zeros((len(omega)+1))
			qplot[:len(qrad)]=qrad-dq
			qplot[len(qrad)] = qrad[-1]+dq
			omegaplot[:len(omega)]=omega-dom
			omegaplot[len(omega)] = omega[-1]+dom
			plt.pcolor(omegaplot,qplot,np.log(Sqrad),vmin=-10,vmax=10)
			plt.colorbar()
			plt.xlabel('omega')
			plt.ylabel('q')
			plt.title('Fourier space ' + kind + ' correlation')
		return qrad, omega,Sqrad
				
		
	# Four point structure factor
	def FourPoint(self,a,qmax=3.14,verbose=True,nmax=20):
		# As written, this thing only works with tracer particles, since I need to track them through the whole simulation
		# Following the template of Wysocki, Winkler, Gompper
		dq=1.0/self.geom.Lx
		nq=int(qmax/dq)
		if nq>nmax:
			print("Coarsening q interval to reduce computational load")
			nq=nmax
			dq=qmax/nq
		nq2=int(2**0.5*nq)
		print("Stepping space Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		FourPoint=np.zeros((nq2,self.Nsnap))
		for u in range(self.Nsnap):
			if (u%10==0):
				print (u)
			smax=self.Nsnap-u
			if self.Nvariable:
				if self.tracer:
					# First filter out the particles we are dealing with: only those that have moved less than distance a
					if self.geom.periodic:
						dists=np.sqrt(np.sum(self.geom.ApplyPeriodic3d(self.rtracers[u:,:,:]-self.rtracers[:smax,:,:])**2,axis=2))
					else:
						dists=np.sqrt(np.sum((self.rtracers[u:,:,:]-self.rtracers[:smax,:,:])**2,axis=2))
					# So now replace those by 1s or 0s depending on how far they have gone
					# this is 0 or 1 as we want, except for negative ones where it's gone far
					hmm=1.0-np.round(dists/a)
					# remove the negative ones
					weights=0.5*(hmm+abs(hmm))
					# Properly: S_4 = 1/N < \sum_n \sum_m w_n(t) w_m(t) e^{iq.(r_n(0)-r_m(0)}>_m,n,t
					# For us this triple sum over m, n and time shift needs to be done here. 
					# Then do the radial averaging in the last step.
					# we can decompose the fourier transform into
					# S_4 = 1/N < \sum_n w_n(t) e^{iq.r_n(0)} \sum_m w_m(t) e^{-iq r_m(0)}>_t
					# So do these first, then radially average, finally take the t average 
					fourierval=np.zeros((nq,nq,smax),dtype=complex)
					fourierrad=np.zeros((nq2,smax),dtype=complex)
					for kx in range(nq):
						for ky in range(nq):
							fourierval[kx,ky,:]=np.sum(weights*np.exp(1j*qx[kx]*self.rtracers[u:,:,0]+1j*qy[ky]*self.rtracers[u:,:,1]),axis=1)
					for l in range(nq2):
						fourierrad[l,:]=np.mean(fourierval[ptsx[l],ptsy[l],:])
					# So now finally multiply and do the shifted average over time. PBC should have been sorted out right in dists? Or not?
					# Should be real at that point
					FourPoint[:,u]=np.real(np.sum(fourierrad*np.conjugate(fourierrad),axis=1))/(self.Ntracer*smax)
				else:
					print("Sorry: Four point function for dividing particles is ambiguous and currently not implemented!")
			else:
				# First filter out the particles we are dealing with: only those that have moved less than distance a
				#print "before distances"
				if self.geom.periodic:
					dists=np.sqrt(np.sum(self.geom.ApplyPeriodic3d(self.rval[u:,:,:]-self.rval[:smax,:,:])**2,axis=2))
				else:
					dists=np.sqrt(np.sum((self.rval[u:,:,:]-self.rval[:smax,:,:])**2,axis=2))
				# So now replace those by 1s or 0s depending on how far they have gone
				# this is 0 or 1 as we want, except for negative ones where it's gone far
				#print "before weights"
				hmm=1.0-np.round(dists/a)
				# remove the negative ones
				weights=0.5*(hmm+abs(hmm))
				# Properly: S_4 = 1/N < \sum_n \sum_m w_n(t) w_m(t) e^{iq.(r_n(0)-r_m(0)}>_m,n,t
				# For us this triple sum over m, n and time shift needs to be done here. 
				# Then do the radial averaging in the last step.
				# we can decompose the fourier transform into
				# S_4 = 1/N < \sum_n w_n(t) e^{iq.r_n(0)} \sum_m w_m(t) e^{-iq r_m(0)}>_t
				# So do these first, then radially average, finally take the t average 
				#print "before Fourier"
				fourierval=np.zeros((nq,nq,smax),dtype=complex)
				fourierrad=np.zeros((nq2,smax),dtype=complex)
				for kx in range(nq):
					for ky in range(nq):
						fourierval[kx,ky,:]=np.sum(weights[:,:]*np.exp(1j*(qx[kx]*self.rval[u:,:,0])+qy[ky]*self.rval[u:,:,1]),axis=1)
				#print "before radial average"
				for l in range(nq2):
					fourierrad[l,:]=np.mean(fourierval[ptsx[l],ptsy[l],:])
				# So now finally multiply and do the shifted average over time. PBC should have been sorted out right in dists? Or not?
				# Should be real at that point
				#print "before fourpoint"
				FourPoint[:,u]=np.real(np.sum(fourierrad*np.conjugate(fourierrad),axis=1))/(self.N*smax)
		tval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			plt.figure()
			vmap=LinearSegmentedColormap('test',cdict,N=nq2) 
			for q in range(0,nq2):
				plt.loglog(tval,FourPoint[q,:],'.-',color=vmap(q))
			plt.xlabel('t')
			plt.ylabel('FourPoint')			
		return tval, FourPoint
	
	
##################################### TO DO: Normal modes ###########################################################
	  
	# Project our displacements or any stuff like that onto the eigenmodes of a hessian matrix, which has been calculated separately
	# we will need self.eigval and self.eigvec
	# I assume that the global skip has already taken care of any of the transient stuff
	# I am *not* removing any dreaded rattlers, because they should be part of the whole thing. 
	def projectModes3d(self,Hessian):
		if self.Nvariable:
			print ("Hessians and dividing particles don't mix! Stopping here!")
			self.proj=0
			self.projv=0
		else:
			# self.rval and self.vval is where the fun is, self.rval=np.zeros((self.Nsnap,self.N,3))
			self.proj=np.zeros((3*Hessian.N,self.Nsnap))
			self.projv=np.zeros((3*Hessian.N,self.Nsnap))
			#proj2=np.zeros((3*Hessian.N,self.Nsnap))
			print (Hessian.eigvec[0:3*Hessian.N:3,0])
			#print Hessian.eigvec[0:3*Hessian.N:3,1]
			# Check out what's going on here
			for u in range(self.Nsnap):
                                # None of this works: either keep it all, or take it all off
                                #takeoff=np.einsum('j,k->jk',np.ones((self.N,)),self.drift[u,:])
                                #rnew=self.rval[u,:,:]-takeoff
                                #ourmean = np.mean(rnew)
                                #hessmean= np.mean(Hessian.rval)
                                #hessnew=Hessian.rval-hessmean+ourmean
                                #print ourmean
                                #print np.mean(hessnew)
				#dr=self.geom.ApplyPeriodic2d(rnew-hessnew)
				dr=self.geom.ApplyPeriodic2d(self.rval[u,:,:]-Hessian.rval)
				dr-=np.mean(dr,axis=0)
				#print np.mean(dr)
				# aah. modulo periodic boundary conditions
				dv=self.vval[u,:,:]
				# serious WTF
				# now project onto the modes
				# This is the organisation of our matrix
				#plt.quiver(self.rval[:,0],self.rval[:,1],self.eigvec[0:3*self.N:3,u],self.eigvec[1:3*self.N:3,u])
				# By definition, we are just ignoring the third dimension here
				self.proj[:,u]=1.0*(np.einsum('i,ij->j',dr[:,0],Hessian.eigvec[0:3*Hessian.N:3,:]) + np.einsum('i,ij->j',dr[:,1],Hessian.eigvec[1:3*Hessian.N:3,:]))
				self.projv[:,u]=1.0*(np.einsum('i,ij->j',dv[:,0],Hessian.eigvec[0:3*Hessian.N:3,:]) + np.einsum('i,ij->j',dv[:,1],Hessian.eigvec[1:3*Hessian.N:3,:]))
				#for v in range(3*Hessian.N):
					#proj2[v,u]=np.sum(dr[:,0]*Hessian.eigvec[0:3*Hessian.N:3,u]) + np.einsum('i,ij->j',dr[:,1],Hessian.eigvec[1:3*Hessian.N:3,:])
			# projection normalization
			self.proj/=self.Nsnap
			self.projv/=self.Nsnap
			self.proj2av=np.sum(self.proj**2,axis=1)
			self.projv2av=np.sum(self.projv**2,axis=1)
			
		return self.proj,self.projv,self.proj2av,self.projv2av
        

        # Project our displacements or any stuff like that onto the eigenmodes of a hessian matrix, which has been calculated separately
	# we will need self.eigval and self.eigvec
	# I assume that the global skip has already taken care of any of the transient stuff
	# I am *not* removing any dreaded rattlers, because they should be part of the whole thing. 
	# Getting rid of boundary particles in the projections
	def projectModes2d(self,Hessian):
		if self.Nvariable:
			print("Hessians and dividing particles don't mix! Stopping here!")
			self.proj=0
			self.projv=0
		else:
			# self.rval and self.vval is where the fun is, self.rval=np.zeros((self.Nsnap,self.N,3))
			self.proj=np.zeros((2*Hessian.N,self.Nsnap))
			self.projv=np.zeros((2*Hessian.N,self.Nsnap))
			#proj2=np.zeros((3*Hessian.N,self.Nsnap))
			#print Hessian.eigvec[0:2*Hessian.N:2,0]
			#print Hessian.eigvec[0:3*Hessian.N:3,1]
			# Check out what's going on here
			transcomp=np.zeros((self.Nsnap,3))
			velcomp=np.zeros((self.Nsnap,3))
			for u in range(self.Nsnap):
				#dr=self.geom.ApplyPeriodic2d(rnew-hessnew)
				if self.geom.periodic:
					dr=self.geom.ApplyPeriodic2d(self.rval[u,:,:]-Hessian.conf.rval)
				else:
					dr = self.rval[u,:,:]-Hessian.conf.rval
				dv=self.vval[u,:,:]
				# now project onto the modes
				# This is the organisation of our matrix
				#plt.quiver(self.rval[:,0],self.rval[:,1],self.eigvec[0:3*self.N:3,u],self.eigvec[1:3*self.N:3,u])
				# By definition, we are just ignoring the third dimension here
				self.proj[:,u]=1.0*(np.einsum('i,ij->j',dr[:,0],Hessian.eigvec[0:2*Hessian.N:2,:]) + np.einsum('i,ij->j',dr[:,1],Hessian.eigvec[1:2*Hessian.N:2,:]))
				self.projv[:,u]=1.0*(np.einsum('i,ij->j',dv[:,0],Hessian.eigvec[0:2*Hessian.N:2,:]) + np.einsum('i,ij->j',dv[:,1],Hessian.eigvec[1:2*Hessian.N:2,:]))
				
			# projection normalization
			self.proj/=self.Nsnap
			self.projv/=self.Nsnap
			self.proj2av=np.sum(self.proj**2,axis=1)
			self.projv2av=np.sum(self.projv**2,axis=1)
			
		return self.proj,self.projv,self.proj2av,self.projv2av,transcomp,velcomp
			
	def plotProjections(self,Hessian,nmodes=5):
	  
		color=iter(cm.rainbow(np.linspace(0,1,nmodes)))
		#multmap=LinearSegmentedColormap('test',cdict,N=nmodes) 
		tval=np.linspace(0,self.Nsnap-1,self.Nsnap)
		plt.figure()
		for u in range(nmodes):
			c=next(color)
			plt.plot(tval,self.proj[u,:],'.-',color=c,label='mode '+str(u))
		if nmodes<10:
			plt.legend()
		plt.xlabel('time')
		plt.ylabel('projection')
		plt.title('Displacements')
		
		color=iter(cm.rainbow(np.linspace(0,1,nmodes)))
		plt.figure()
		for u in range(nmodes):
			c=next(color)
			plt.plot(tval,self.projv[u,:],'.-',color=c,label='mode '+str(u))
		if nmodes<10:
			plt.legend()
		plt.xlabel('time')
		plt.ylabel('projection')
		plt.title('Velocities')
		  
		
		omega = np.sqrt(np.abs(Hessian.eigval))
		plt.figure()
		plt.plot(omega,0.5*Hessian.eigval*self.proj2av,'.-r',label='projection')
		plt.xlabel(r'$\omega$')
		plt.ylabel('Energy')
		plt.xlim(0,12)
		#plt.ylim(1e-10,1e-5)
		plt.title('Energy from displacement projections')
		
		plt.figure()
		plt.plot(omega,self.proj2av/np.sum(self.proj2av),'.-r',label='displacements')
		plt.xlabel(r'$\omega$')
		plt.ylabel('projection square')
		#plt.xlim(-0.01,12)
		plt.legend()
		plt.title('Square projections (normalised)')
		
		plt.figure()
		plt.plot(omega,self.projv2av/np.sum(self.projv2av),'.-g',label='velocities')
		plt.xlabel(r'$\omega$')
		plt.ylabel('projection square')
		#plt.xlim(-0.01,12)
		plt.legend()
		plt.title('Square projections (normalised)')
		
		
		
		
			
	


