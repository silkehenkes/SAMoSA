
from Configuration import *
from Hessian import *

# Dynamics is a child class of configuration
# All its methods are dynamical, i.e. they include some time-correlated information	
class Dynamics(Configuration):
	
	# Use the Configuration default constructor. We *need* to be set to self.multiopt == "many"!
	def validate_initialise():
		if not self.multiopt == "many":
			print("Dynamics:: Error: Cannot use dynamics without reading in multiple files! Aborting.")
			break
		# Some booleans to avoid redoing the whole shebang twice as some computations depend on others
		self.hasMSD = False
		# Have we isolated the particles we are following? Implicit assumption: we are only ever tracking the 
		# same types of particles in a given instance.
		# If a problem, can always set it to 'False' again from outside
		self.tracked = False
	
	# By design, this is only meaningful on the whole data set, e.g. do not subtract for tracer particles only
	def takeDrift()
		self.drift=np.zeros((self.Nsnap,3))
		if self.Nvariable:
			print("Dynamics:: Variable N: Taking off the drift is meaningless. Doing nothing.")
		else:	
			for u in range(1,self.Nsnap):
				dr=self.geom.ApplyPeriodic2d(self.rval[u,:,:]-self.rval[u-1,:,:])
				drift0=np.sum(dr,axis=0)/self.N
				self.drift[u,:]=self.drift[u-1,:]+drift0
				#print self.drift[u,:]
			
	# Tracking a subset of particles: This identifies the relevant ones
	# and gives an error if that number changes or the flags don't match
	# complicated: if the particles we are aiming to track move around in their list index
	def self.getTrack(usetype):
		if usetype == 'all':
			if self.Nvariable:
				print("Dynamics::getTrack - Attempting to track ALL particles while N is variable: this doesn't make sense. Stopping")
				break
			else:
				self.Ntrack = self.N
				self.usethese = range(self.Ntrack)
				self.complicated = False
		else:
			useparts = self.getUseparts(usetype,0)
			flag0 = self.flag(0,useparts)
			self.Ntrack = len(useparts)
			usethese0 = np.empty((self.Nsnap,self.Ntrack))
			# If the positions of our flagged particles are jumping around, we are in deep doodoo for vectorisation
			self.complicated = False
			for u in range(0,self.Nsnap):
				useparts = self.getUseparts(usetype,u)
				N = len(useparts)
				if N != self.Ntrack:
					print("Dynamics::getTrack - Error: Number of tracked particles is changing, check that chosen type is not dividing or dying!")
					break
				flag = self.flag(u,useparts)
				# Check that 1. These are the same labels at the same places. Can continue to use vectorisation in that case
				if not np.array.equal(flag,flag0):
					self.complicated = True
					# However, they could still be there, but relabeled. Then things are non-vectorisable, but possible, though slow
					if set(flag) == set(flag0):
						# now we need to get useparts in the same order as the initial flags
						# Better slow than wrong ...
						for k in flag:
							idx = flag0.index(k)
							usethese0[u,idx] = useparts[k]
					else:
						print("Dynamics::getTrack - Error: Labels (flags) of tracked particles are changing, something is wrong!")
						break
			if self.complicated:
				self.usethese = usethese0
			else:
				self.usethese = useparts
				
				
	# relative velocity distribution (and average velocity)
	# component wise as well, assumes x and y directions only
	def getVelDist(self,usetype='all',bins,bins2):
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
		return vav, vdist,vdist2		
		
####################################### Glassy Physics ###############################################################
	def getMSD(self,usetype='all',takeDrift,verbose=True):
		self.msd=np.empty((self.Nsnap,))
		
		# Careful, we need a sanity check: We can only compute a MSD if the particles we're tracking isn't changing (whatever the rest of the system is doing)
		if not self.tracked:
			self.getTrack(usetype)				
		
		if takeDrift:
			self.takeDrift()
			
		for u in range(self.Nsnap):	
			smax=self.Nsnap-u
			# Note that by design, take drift needs to work on the whole data set
			if takeDrift:
				if self.complicated:
					print("Dynamics::getMSD - Taking off drift at constant N while complicated labels: My head is hurting, stopping here. Reconsider what you're doing ... it's probably wrong")
					break
				else:
					hmm=(self.drift[:smax,:]-self.drift[u:,:])
					takeoff=np.einsum('j,ik->ijk',np.ones((self.Ntrack,)),hmm)
				if self.geom.periodic:
					dr=self.geom.ApplyPeriodic3d(self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:])-takeoff[self.usethese,:]
				else:
					dr=self.rval[:smax,self.usethese,:]-self.rval[u:,self.usethese,:]-takeoff[self.usethese,:]
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
			
			#plt.show()
		
		return xval, self.msd
            
	# Velocity autocorrelation function
	def getVelAuto(self,usetype='all',verbose=True):
		self.velauto=np.empty((self.Nsnap,))
		v2av=np.empty((self.Nsnap,))
		
		if not self.tracked:
			self.getTrack(usetype)
			
		# First compute normalised velocities. Note: normalised by mean velocity in the whole system at that time, not unit vectors!
		vnormed = np.zeros((self.Nsnap,self.Ntrack,3))
		for u in range(self.Nsnap):
			if self.complicated:
				v2av[u]=np.sum(np.sum((self.vval[u,list(self.usethese[u,:]),:])**2,axis=1),axis=0)/(self.Ntrack)
				vnormed[u,:,:]=self.vval[u,list(self.usethese[u,:]),:]/np.sqrt(v2av[u])
			else:
				v2av[u]=np.sum(np.sum((self.vval[u,self.usethese,:])**2,axis=1),axis=0)/(self.Ntrack)
				vnormed[u,:,:]=self.vval[u,self.usethese,:]/np.sqrt(v2av[u])
				
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			if self.complicated:
				print("Dynamics::getVelAuto - Warning: Complicated relabeling: This is slow, make sure to only track a few particles") 
				for s in range(smax):
					self.velauto[u] += np.sum(vnormed[s,list(self.usethese[s,:]),0]*vnormed[u+s,list(self.usethese[s,:]),0]+vnormed[s,list(self.usethese[s,:]),1]*vnormed[u+s,list(self.usethese[s,:]),1]+vnormed[s,list(self.usethese[s,:]),2]*vnormed[u+s,list(self.usethese[s,:]),2])/(self.Ntrack*smax)
			else:
				self.velauto[u]=np.sum(np.sum((vnormed[:smax,self.usethese,0]*vnormed[u:,self.usethese,0]+vnormed[:smax,self.usethese,1]*vnormed[u:,self.usethese,1]+vnormed[:smax,self.usethese,2]*vnormed[u:,self.usethese,2]),axis=1),axis=0)/(self.Ntrack*smax)

                                
		xval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		if verbose:
			fig=plt.figure()
			plt.loglog(xval,self.velauto,'r.-',lw=2)
			plt.xlabel('time')
			plt.ylabel('Velocity autocorrelation')
			
			#plt.show()
		return xval, self.velauto, v2av
            
        
	# Computes non-gaussian factor / kurtosis (4th moment), and also the MSD (2nd moment) if it hasn't been done yet
	# returns all three
	def getNonGaussian(self,usetype='all',takeDrift,verbose=True):
		
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
			if takeDrift:
				if self.complicated:
					print("Dynamics::getNonGaussian - Taking off drift at constant N while complicated labels: My head is hurting, stopping here. Reconsider what you're doing ... it's probably wrong")
					break
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
	def SelfIntermediate(self,qval,usetype='all',verbose=True):
		# This is single particle, single q, shifted time step. Equivalent to the MSD, really
		SelfInt=np.empty((self.Nsnap,),dtype=complex)
		
		if not self.tracked:
			self.getTrack(usetype)
			
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			if takeDrift:
				if self.complicated:
					print("Dynamics::SelfIntermediate - Taking off drift at constant N while complicated labels: My head is hurting, stopping here. Reconsider what you're doing ... it's probably wrong")
					break
				else:
					hmm=(self.drift[:smax,:]-self.drift[u:,:])
					takeoff=np.einsum('j,ik->ijk',np.ones((self.Ntrack,)),hmm)
				if self.geom.periodic:
					SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(self.geom.ApplyPeriodicX(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0])+takeoff[:,self.usethese,0])+1.0j*qval[1]*(self.geom.ApplyPeriodicY(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1])+takeoff[:,self.usethese,1])+1.0j*qval[2]*(self.geom.ApplyPeriodicZ(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2])+takeoff[:,self.usethese,2])),axis=1),axis=0)/(self.Ntrack*smax)
				else:
					SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0]+takeoff[:,self.usethese,0])+1.0j*qval[1]*(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1]+takeoff[:,self.usethese,1])+1.0j*qval[2]*(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2]+takeoff[:,self.usethese,2])),axis=1),axis=0)/(self.Ntrack*smax)
			else:
				if complicated:
					for s in range(smax):
						ps = list(self.usethese[s,:])
						pu = list(self.usethese[u+s,:])
						if self.geom.periodic:
							SelfInt[u] += np.sum(np.exp(1.0j*qval[0]*(self.geom.ApplyPeriodicX(-self.rval[s,ps,0]+self.rval[u+s,pu,0]))+1.0j*qval[1]*(self.geom.ApplyPeriodicY(-self.rval[s,ps,1]+self.rval[u+s,pu,1]))+1.0j*qval[2]*(self.geom.ApplyPeriodicZ(-self.rval[s,ps,2]+self.rval[u+s,pu,2]))))/(self.N*smax)
						else:
							SelfInt[u] += np.sum(np.sum(np.exp(1.0j*qval[0]*(-self.rval[s,ps,0]+self.rval[u+2,pu,0])+1.0j*qval[1]*(-self.rval[s,ps,1]+self.rval[u+s,pu,1])+1.0j*qval[2]*(-self.rval[s,ps,2]+self.rval[u+s,pu,2])))/(self.Ntrack*smax)
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
		
	
#################### Not checked, probably buggy / physics wrong frome since forever ###############  
	# Well, that seems to do fuck all
	def getDynStruct(self,qmax,omegamax,verbose=True,nmax=50):
		# Following the template of Wysocki, Winkler, Gompper
		dq=1.0/self.geom.Lx
		nq=int(qmax/dq)
		if nq>nmax:
			print "Coarsening q interval to reduce computational load"
			nq=nmax
			dq=qmax/nq
		nq2=int(2**0.5*nq)
		print "Stepping space Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		dom=1.0/(self.Nsnap*self.param.dt*self.param.dump['freq'])
		nom1=int(omegamax/dom)
		nom=2*int(omegamax/dom)+1
		print "Stepping time Fourier transform with step " + str(dom)+ ", resulting in " + str(nom)+ " steps."
		# Formally: S(q,omega) = 1/N int dt \rho_q(t) \rho*_q(0) e^i\omega t, where \rho_q(t) = \int dr \rho(r,t) e^iq r
		# The second part is what we already had for the positional static structure factor
		# For simplicity reasons, do the radial averaging before taking the time transform
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		rhorad=np.zeros((self.Nsnap,nq2),dtype=complex)
		for u in range(self.Nsnap):
			if (u%10==0):
				print u
			fourierval=np.empty((nq,nq),dtype=complex)
			for kx in range(nq):
				for ky in range(nq):
					# And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
					fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.rval[u,:,0]+qy[ky]*self.rval[u,:,1])))
			for l in range(nq2):
				rhorad[u,l]=np.mean(fourierval[ptsx[l],ptsy[l]])
		# Do our little shifted averaging procedure at constant q now
		rhocorr=np.zeros((self.Nsnap,nq2),dtype=complex)
		for u in range(self.Nsnap):
			smax=self.Nsnap-u
			rhocorr[u,:]=np.sum(rhorad[u:,:]*rhorad[:smax,:],axis=0)/smax
		# Cute. Now do the time tranform:
		DynStruct=np.zeros((nom,nq2),dtype=complex)
		tval=np.linspace(0,self.Nsnap*self.param.dt*self.param.dump['freq'],num=self.Nsnap)
		omega=np.empty((nom,))
		for no in range(0,nom):
			omega[no]=(-nom1+no)*dom
			DynStruct[no,:]=np.einsum('ij,i', rhocorr, np.exp(1j*omega[no]*tval))
		print omega
		# OK, what have we got? Take the absolute value and look
		PlotDynStruct=np.real(DynStruct)**2+np.imag(DynStruct)**2
		if verbose:
			plt.figure()
			plt.pcolor(qrad,omega,np.log10(PlotDynStruct))
			plt.colorbar()
			plt.title('Dynamical structure factor')
			
			plt.figure()
			plt.pcolor(qrad,tval,np.log10(np.real(rhocorr)))
			plt.colorbar()
			plt.title('Density correlation function')
		if verbose:
			plt.figure()
			plt.plot(omega,np.log10(PlotDynStruct[:,0]),'.-k')
			plt.plot(omega,np.log10(PlotDynStruct[:,1]),'.-r')
			plt.plot(omega,np.log10(PlotDynStruct[:,5]),'.-g')
			plt.plot(omega,np.log10(PlotDynStruct[:,10]),'.-b')
			plt.xlabel('omega')
			plt.ylabel('structure factor')
			plt.title('Dynamical structure factor')

			
			plt.figure()
			plt.plot(tval,np.log10(rhocorr[:,0]),'.-k')
			plt.plot(tval,np.log10(rhocorr[:,1]),'.-g')
			plt.plot(tval,np.log10(rhocorr[:,5]),'.-r')
			plt.plot(tval,np.log10(rhocorr[:,10]),'.-b')
			plt.title('Density correlation function')
		return omega,qrad,DynStruct
		
		
	# Four point structure factor
	def FourPoint(self,a,qmax=3.14,verbose=True,nmax=20):
		# As written, this thing only works with tracer particles, since I need to track them through the whole simulation
		# Following the template of Wysocki, Winkler, Gompper
		dq=1.0/self.geom.Lx
		nq=int(qmax/dq)
		if nq>nmax:
			print "Coarsening q interval to reduce computational load"
			nq=nmax
			dq=qmax/nq
		nq2=int(2**0.5*nq)
		print "Stepping space Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		FourPoint=np.zeros((nq2,self.Nsnap))
		for u in range(self.Nsnap):
			if (u%10==0):
				print u
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
					print "Sorry: Four point function for dividing particles is ambiguous and currently not implemented!"
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
	def projectModes(self,Hessian):
		if self.Nvariable:
			print "Hessians and dividing particles don't mix! Stopping here!"
			self.proj=0
			self.projv=0
		else:
			# self.rval and self.vval is where the fun is, self.rval=np.zeros((self.Nsnap,self.N,3))
			self.proj=np.zeros((3*Hessian.N,self.Nsnap))
			self.projv=np.zeros((3*Hessian.N,self.Nsnap))
			#proj2=np.zeros((3*Hessian.N,self.Nsnap))
			print Hessian.eigvec[0:3*Hessian.N:3,0]
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
	def projectModes2d(self,Hessian):
		if self.Nvariable:
			print "Hessians and dividing particles don't mix! Stopping here!"
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
				transcomp[u,:]=np.mean(dr,axis=0)
				dr-=np.mean(dr,axis=0)
				#print np.mean(dr)
				# aah. modulo periodic boundary conditions
				dv=self.vval[u,:,:]
				velcomp[u,:]=np.mean(dv,axis=0)
				# serious WTF
				# now project onto the modes
				# This is the organisation of our matrix
				#plt.quiver(self.rval[:,0],self.rval[:,1],self.eigvec[0:3*self.N:3,u],self.eigvec[1:3*self.N:3,u])
				# By definition, we are just ignoring the third dimension here
				self.proj[:,u]=1.0*(np.einsum('i,ij->j',dr[:,0],Hessian.eigvec[0:2*Hessian.N:2,:]) + np.einsum('i,ij->j',dr[:,1],Hessian.eigvec[1:2*Hessian.N:2,:]))
				self.projv[:,u]=1.0*(np.einsum('i,ij->j',dv[:,0],Hessian.eigvec[0:2*Hessian.N:2,:]) + np.einsum('i,ij->j',dv[:,1],Hessian.eigvec[1:2*Hessian.N:2,:]))
				#for v in range(3*Hessian.N):
					#proj2[v,u]=np.sum(dr[:,0]*Hessian.eigvec[0:3*Hessian.N:3,u]) + np.einsum('i,ij->j',dr[:,1],Hessian.eigvec[1:3*Hessian.N:3,:])
			# projection normalization
			self.proj/=self.Nsnap
			self.projv/=self.Nsnap
			self.proj2av=np.sum(self.proj**2,axis=1)
			self.projv2av=np.sum(self.projv**2,axis=1)
			
		return self.proj,self.projv,self.proj2av,self.projv2av,transcomp,velcomp
			
	def plotProjections(self,Hessian,nmodes=5):
	  
		multmap=LinearSegmentedColormap('test',cdict,N=nmodes) 
		tval=np.linspace(0,self.Nsnap-1,self.Nsnap)
		plt.figure()
		for u in range(nmodes):
			plt.plot(tval,self.proj[u,:],'.-',color=multmap(u),label='mode '+str(u))
		plt.legend()
		plt.xlabel('time')
		plt.ylabel('projection')
		plt.title('Displacements')
		
		plt.figure()
		for u in range(nmodes):
			plt.plot(tval,self.projv[u,:],'.-',color=multmap(u),label='mode '+str(u))
		plt.legend()
		plt.xlabel('time')
		plt.ylabel('projection')
		plt.title('Velocities')
		  
		
		tau=1.0/float(self.param.nu)
		v=float(self.param.v0)
		
		plt.figure()
		plt.loglog(Hessian.eigval,0.5*Hessian.eigval*self.proj2av,'.-r',label='projection')
		# directly add the prediction here
		plt.loglog(Hessian.eigval,v**2*tau/(4.0*(1.0+Hessian.eigval*tau)),'-k',label='prediction')
		plt.xlabel(r'$\lambda$')
		plt.ylabel('Energy')
		plt.xlim(0,12)
		#plt.ylim(1e-10,1e-5)
		plt.title('Energy from displacement projections, v0=' + str(self.param.v0)+ ' tau=' + str(tau))
		
		plt.figure()
		plt.loglog(Hessian.eigval,self.proj2av,'o-r',label='displacements')
		plt.loglog(Hessian.eigval,self.projv2av,'o-g',label='velocities')
		plt.xlabel(r'$\lambda$')
		plt.ylabel('projection square')
		#plt.xlim(-0.01,12)
		plt.legend()
		plt.title('Square projections, v0=' + str(self.param.v0)+ ' tau=' + str(tau))
		
		
		plt.figure()
		plt.semilogy(Hessian.eigval,Hessian.eigval**2*self.proj2av,'o-r',label='displacements')
		plt.semilogy(Hessian.eigval,self.projv2av,'o-g',label='velocitites')
		plt.xlabel(r'$\lambda$')
		plt.ylabel('velocity projections (expected)')
		plt.xlim(-0.01,12)
		plt.legend()
		plt.title('Square velocity projections, v0=' + str(self.param.v0)+ ' tau=' + str(tau))
		
		
			
	


