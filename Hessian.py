

from Configuration import *
from numpy import linalg as LA

class Hessian:
    # passing conf here to construct matrix
    # Make sure that this already contains only the particles that one wants
	def __init__(self,conf,rattlers=[],debug=False):
		# First check for suitable data format of configuration
		if conf.multiopt == "many":
			print("Hessian: Error - Can only construct a Hessian with a single set of input coordinates. Construct single-position configuration instead.")
			break
		self.conf=conf
		self.geom = conf.geom
		#self.rval=self.conf.rval
		self.rattlers=rattlers
		self.N=self.conf.N
		self.Nrigid=self.N-len(rattlers)
		self.geom=geom
		#self.inter=self.conf.inter
		self.debug=debug
	
	def setRattlers(rattlers):
		self.rattlers = rattlers
		self.Nrigid=self.N-len(rattlers)
        
	
	# This needs to be one of the single-frame configurations, create check for it
	def makeMatrix(self,dim=2,addRestoring=True,ksurf=10.0):
		# Dimensions of the problem
		self.dim = dim
		# Check what dimension we are after
		if dim == 3:
			addCurvature=False
			if (self.geom.manifold=='sphere'):
				addCurvature=True
				Rval=self.geom.R
				print("Hessian: Calculating 3d Hessian on a sphere!")
			elif (self.geom.manifold=='plane'):
				print("Hessian: Calculating 3d Hessian on a plane!")
			else:
				print("Hessian: Error: 3d Hessian has not yet been implemented on " + self.geom.manifold + " manifolds!")
				break
			# This matrix is in principle 3N by 3N. We will have to be careful later on in throwing out extra off-surface modes
			print("Hessian: Info - allocating the " + str(3*self.conf.N) + " by " + str(3*self.conf.N) + " 3d Hessian matrix.")
			self.Hessian=np.zeros((3*self.Nrigid,3*self.N))
			# The unit normal for everybody will be used for the transverse restoring forces
			Normal=self.geom.UnitNormal(self.conf.rval)
		elif dim == 2:
			if (self.geom.manifold=='plane'):
				print("Hessian: Calculating Hessian on a plane!")
			else:
				print("Hessian: Error: 2d Hessian has not yet been implemented on " + self.geom.manifold + " manifolds!" )
				break
			print("Hessian: Info - allocating the " + str(2*self.N) + " by " + str(2*self.N) + " 2d Hessian matrix.")
			self.Hessian=np.zeros((2*self.Nrigid,2*self.N))
		else:
			print("Hessian: Error - unknown dimension argument " + str(dim) + " is not 2 or 3. Stopping.")
			break
		# Constructing the actual matrix
		fsum=0.0
		fav=0.0
		for i in range(self.N):
			if i not in self.rattlers:
				if (i%200==0):
					print i
				# get some of the constants that are necessary here:
				neighbours, drvec, radi, radj = self.conf.getNeighbours(i,self.conf.inter.getMult(),self.conf.inter.getDmax())
				# particle distances and contact normal vectors
				dr =np.sqrt(drvec[:,0]**2+drvec[:,1]**2+drvec[:,2]**2)
				nij=np.transpose(np.transpose(drvec)/np.transpose(dr))
				# Forces
				fvec=self.conf.inter.getForce(i,neighbours,drvec,radi,radj)
				fsum+=sum(fvec)
				# Projected onto the contact normal
				fval=np.sum(fvec*nij,axis=1)
				fav+=sum(fval)
				# Stiffnesses
				kij=self.conf.inter.getStiffness(i,neighbours,drvec,radi,radj)
				# equilibrium distances are given by dr already
				
				# Now we are filling the matrix in either 3d or 2d
				if dim == 3:
					# Alright: elements are labeled as follows: Contact ij has sub-square 3i, 3i+1, 3i+2 and 3j, 3j+1, 3j+2
					diagsquare=np.zeros((3,3))
					for j in range(len(neighbours)):
						n=nij[j,:]
						N=Normal[neighbours[j],:]
						subsquare=np.zeros((3,3))
						# xx, xy and xz
						subsquare[0,0]=-fval[j]/dr[j]*(1-n[0]*n[0]-N[0]*N[0])+kij[j]*n[0]*n[0]
						subsquare[0,1]=-fval[j]/dr[j]*(0-n[0]*n[1]-N[0]*N[1])+kij[j]*n[0]*n[1]
						subsquare[0,2]=-fval[j]/dr[j]*(0-n[0]*n[2]-N[0]*N[2])+kij[j]*n[0]*n[2]
						# yx, yy and yz
						subsquare[1,0]=-fval[j]/dr[j]*(0-n[1]*n[0]-N[1]*N[0])+kij[j]*n[1]*n[0]
						subsquare[1,1]=-fval[j]/dr[j]*(1-n[1]*n[1]-N[1]*N[1])+kij[j]*n[1]*n[1]
						subsquare[1,2]=-fval[j]/dr[j]*(0-n[1]*n[2]-N[1]*N[2])+kij[j]*n[1]*n[2]
						# zx, zy and zz
						subsquare[2,0]=-fval[j]/dr[j]*(0-n[2]*n[0]-N[2]*N[0])+kij[j]*n[2]*n[0]
						subsquare[2,1]=-fval[j]/dr[j]*(0-n[2]*n[1]-N[2]*N[1])+kij[j]*n[2]*n[1]
						subsquare[2,2]=-fval[j]/dr[j]*(1-n[2]*n[2]-N[2]*N[2])+kij[j]*n[2]*n[2]
						# Stick into the big matrix
						label=neighbours[j]
						self.Hessian[3*i:(3*i+3),3*label:(3*label+3)]=-subsquare
						# Add the required bits to the diagonal part of the matrix
						# xx, xy and xz
						diagsquare[0,0]+=fval[j]/dr[j]*(1-n[0]*n[0]-N[0]*N[0])-kij[j]*n[0]*n[0]
						diagsquare[0,1]+=fval[j]/dr[j]*(0-n[0]*n[1]-N[0]*N[1])-kij[j]*n[0]*n[1]
						diagsquare[0,2]+=fval[j]/dr[j]*(0-n[0]*n[2]-N[0]*N[2])-kij[j]*n[0]*n[2]
						# yx, yy and yz
						diagsquare[1,0]+=fval[j]/dr[j]*(0-n[1]*n[0]-N[1]*N[0])-kij[j]*n[1]*n[0]
						diagsquare[1,1]+=fval[j]/dr[j]*(1-n[1]*n[1]-N[1]*N[1])-kij[j]*n[1]*n[1]
						diagsquare[1,2]+=fval[j]/dr[j]*(0-n[1]*n[2]-N[1]*N[2])-kij[j]*n[1]*n[2]
						# zx, zy and zz
						diagsquare[2,0]+=fval[j]/dr[j]*(0-n[2]*n[0]-N[2]*N[0])-kij[j]*n[2]*n[0]
						diagsquare[2,1]+=fval[j]/dr[j]*(0-n[2]*n[1]-N[2]*N[1])-kij[j]*n[2]*n[1]
						diagsquare[2,2]+=fval[j]/dr[j]*(1-n[2]*n[2]-N[2]*N[2])-kij[j]*n[2]*n[2]
						# Add the curvature term if required
						# Sooooo. The derivation says there is just a -fval/Rval n N term here, due to the tilt of the normal with parallel transport
						# However, this term very explicitly punishes out-from-surface deviations. 
						# Making this larger gives much cleaner results. Soo - huh?
						if (addCurvature):
							# xx, xy and xz
							diagsquare[0,0]+=-fval[j]/Rval*n[0]*N[0]
							diagsquare[0,1]+=-fval[j]/Rval*n[1]*N[0]
							diagsquare[0,2]+=-fval[j]/Rval*n[2]*N[0]
							# yx, yy and yz
							diagsquare[1,0]+=-fval[j]/Rval*n[0]*N[1]
							diagsquare[1,1]+=-fval[j]/Rval*n[1]*N[1]
							diagsquare[1,2]+=-fval[j]/Rval*n[2]*N[1]
							# zx, zy and zz
							diagsquare[2,0]+=-fval[j]/Rval*n[0]*N[2]
							diagsquare[2,1]+=-fval[j]/Rval*n[1]*N[2]
							diagsquare[2,2]+=-fval[j]/Rval*n[2]*N[2]
						
						if (addRestoring):
							# Manual restoring force along the normal
							#ksurf=10
							diagsquare[0,0]+=-ksurf*N[0]*N[0]
							diagsquare[0,1]+=-ksurf*N[1]*N[0]
							diagsquare[0,2]+=-ksurf*N[2]*N[0]
							# yx, yy and yz
							diagsquare[1,0]+=-ksurf*N[0]*N[1]
							diagsquare[1,1]+=-ksurf*N[1]*N[1]
							diagsquare[1,2]+=-ksurf*N[2]*N[1]
							# zx, zy and zz
							diagsquare[2,0]+=-ksurf*N[0]*N[2]
							diagsquare[2,1]+=-ksurf*N[1]*N[2]
							diagsquare[2,2]+=-ksurf*N[2]*N[2]
					#print diagsquare
					self.Hessian[3*i:(3*i+3),3*i:(3*i+3)]=-diagsquare
				else:
					diagsquare=np.zeros((2,2))
					for j in range(len(neighbours)):
						n=nij[j,:]
						N=Normal[neighbours[j],:]
						subsquare=np.zeros((2,2))
						# xx, xy and xz
						subsquare[0,0]=-fval[j]/dr[j]*(1-n[0]*n[0])+kij[j]*n[0]*n[0]
						subsquare[0,1]=-fval[j]/dr[j]*(0-n[0]*n[1])+kij[j]*n[0]*n[1]
						# yx, yy and yz
						subsquare[1,0]=-fval[j]/dr[j]*(0-n[1]*n[0])+kij[j]*n[1]*n[0]
						subsquare[1,1]=-fval[j]/dr[j]*(1-n[1]*n[1])+kij[j]*n[1]*n[1]
						# Stick into the big matrix
						label=neighbours[j]
						self.Hessian[2*i:(2*i+2),2*label:(2*label+2)]=-subsquare
						# Add the required bits to the diagonal part of the matrix
						# xx, xy and xz
						diagsquare[0,0]+=fval[j]/dr[j]*(1-n[0]*n[0])-kij[j]*n[0]*n[0]
						diagsquare[0,1]+=fval[j]/dr[j]*(0-n[0]*n[1])-kij[j]*n[0]*n[1]
						# yx, yy and yz
						diagsquare[1,0]+=fval[j]/dr[j]*(0-n[1]*n[0])-kij[j]*n[1]*n[0]
						diagsquare[1,1]+=fval[j]/dr[j]*(1-n[1]*n[1])-kij[j]*n[1]*n[1]
					#print diagsquare
					self.Hessian[2*i:(2*i+2),2*i:(2*i+2)]=-diagsquare
		fav/=self.N
		print "Hessian: Estimating distance from mechanical equilibrium of initial configuration "
		print "Scaled force sum is " + str(fsum/fav)
			
	def getModes(self):
		# Let's have a look if what we get is in any way reasonable
		# Eigenvalues and eigenvectors
		# Only symmetrise to calculate - for clarity and debugging above
		HessianSym=0.5*(self.Hessian+np.transpose(self.Hessian))
		if self.debug:
			plt.figure()
			plt.pcolor(HessianSym)
		#HessianASym=0.5*(self.Hessian-np.transpose(self.Hessian))
		#print HessianASym
		# Use routines for hermitian eigenvector decomposition
		# Default is ascending order, which suits us
		print "Starting Diagonalisation!"
		self.eigval, self.eigvec = LA.eigh(HessianSym)
		print "The smallest eigenvalue is: " + str(np.amin(self.eigval))
		print self.eigval
		# Crosscheck on sanity
		if self.dim == 3:
			wx=np.zeros((self.N,))
			wy=np.zeros((self.N,))
			wz=np.zeros((self.N,))
			for u in range(self.N):
					# dimensional contributions
					wx[u]=np.sum(self.eigvec[0:3*self.N:3,u]**2)
					wy[u]=np.sum(self.eigvec[1:3*self.N:3,u]**2)
					wz[u]=np.sum(self.eigvec[2:3*self.N:3,u]**2)
			if self.debug:
				# start with some debugging output
				plt.figure()
				eigrank=np.linspace(0,3*self.N,3*self.N)
				plt.plot(eigrank,self.eigval,'.-')
				print wx
				print wy
				print wz
			return wx,wy,wz
		else:
			wx=np.zeros((self.N,))
			wy=np.zeros((self.N,))
			for u in range(self.N):
					# dimensional contributions
					wx[u]=np.sum(self.eigvec[0:2*self.N:2,u]**2)
					wy[u]=np.sum(self.eigvec[1:2*self.N:2,u]**2)
			if self.debug:
				# start with some debugging output
				plt.figure()
				eigrank=np.linspace(0,2*self.N,2*self.N)
				plt.plot(eigrank,self.eigval,'.-')
				print wx
				print wy
			return wx,wy
			
		
	def plotModes3d(self,omegamax=3.0,npts=100):
		if self.dim != 3:
			print("Hessian: Error - Modes plotting is currently implemented in 3d only")
			break
		# Straight here: The projection ratios on the sphere/plane
		projrat=np.zeros((3*self.N,))
		# Get the tangent bundle and coordinates
		# Note: this works for anything! In the plane, these are x, y, ex, ey
		theta,phi,etheta,ephi=self.geom.TangentBundle(self.conf.rval)
		for u in range(3*self.N):
			thproj=etheta[:,0]*self.eigvec[0:3*self.N:3,u]+etheta[:,1]*self.eigvec[1:3*self.N:3,u]+etheta[:,2]*self.eigvec[2:3*self.N:3,u]
			phiproj=ephi[:,0]*self.eigvec[0:3*self.N:3,u]+ephi[:,1]*self.eigvec[1:3*self.N:3,u]+ephi[:,2]*self.eigvec[2:3*self.N:3,u]
			projrat[u]=np.sum(thproj**2)+np.sum(phiproj**2)	
		# For this histogram and only this histogram: replace negative eigenvalues by zeros
		eigplot=self.eigval
		badlist=list(np.where(eigplot<0.0)[0])
		eigplot[badlist]=0.0
		omega=np.real(np.sqrt(eigplot))
		ombin=np.linspace(0,omegamax,npts)
		dom=ombin[1]-ombin[0]
		omhist, bin_edges = np.histogram(omega,bins=ombin)
		omlabel=(np.round(omega/dom)).astype(int)
		projhist=np.zeros((npts-1,))
		projcount=np.zeros((npts-1,))
		for l in range(npts-1):
			pts=np.nonzero(omlabel==l)[0]
			projhist[l]+=sum(projrat[pts])
			projcount[l]+=len(pts)
		isdata=[index for index, value in enumerate(projcount) if value>0]
		projhist[isdata]/=projcount[isdata]
					
		plt.figure()
		plt.plot(ombin[1:]-dom/2,omhist,'o-k',label='DOS')
		plt.xlim(0,omegamax)
		plt.xlabel('omega')
		plt.ylabel('D(omega)')
		plt.title('Density of states')
		
		plt.figure()
		plt.plot(ombin[isdata]+dom/2,projhist[isdata],'o-r',label='Surface projection')
		plt.ylim(0,1.1)
		plt.xlim(0,omegamax)
		plt.xlabel('omega')
		plt.ylabel('projection')
		plt.title('Surface projection value')
		
		# Plotting eigenvectors (#0, #N and #2N)
		if self.geom.manifold=='plane':
			usepts=[0,1,2,3,self.N,2*self.N-1,3*self.N-1]
			for u in usepts:
				plt.figure()
				plt.quiver(self.conf.rval[:,0],self.conf.rval[:,1],self.eigvec[0:3*self.N:3,u],self.eigvec[1:3*self.N:3,u])
				# dimensional contributions
				wx=np.sum(self.eigvec[0:3*self.N:3,u]**2)
				wy=np.sum(self.eigvec[1:3*self.N:3,u]**2)
				wz=np.sum(self.eigvec[2:3*self.N:3,u]**2)
				plt.title('Eigenvector #' + str(u) + ' ' + str(wx+wy) + ' on plane')
				
		if self.geom.manifold=='sphere':
			# Get the tangent bundle and coordinates
			theta,phi,etheta,ephi=self.geom.TangentBundle(self.rval)
			usepts=[0,1,2,3,self.N,2*self.N-1,3*self.N-1]
			for u in usepts:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.quiver(self.conf.rval[:,0], self.conf.rval[:,1], self.conf.rval[:,2], self.eigvec[0:3*self.N:3,u], self.eigvec[1:3*self.N:3,u], self.eigvec[2:3*self.N:3,u])
				# See how much of the mode is along the sphere here
				thproj=etheta[:,0]*self.eigvec[0:3*self.N:3,u]+etheta[:,1]*self.eigvec[1:3*self.N:3,u]+etheta[:,2]*self.eigvec[2:3*self.N:3,u]
				phiproj=ephi[:,0]*self.eigvec[0:3*self.N:3,u]+ephi[:,1]*self.eigvec[1:3*self.N:3,u]+ephi[:,2]*self.eigvec[2:3*self.N:3,u]
				frac=np.sum(thproj**2)+np.sum(phiproj**2)
				plt.title('Eigenvector #' + str(u) + ' ' + str(frac) +' on sphere') 
		
	# Same helper function as in configuration for radially averaged Fourier transforms
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
		
	# project the modes into Fourier space to see how it's scaling
	def ModesFourierLongTrans(self,whichmode,qmax=0.3,verbose=True,eps=0.001):
		# Strictly 2d affair
		if self.dim != 2:
			print("Error: Hessial - Fourier transform is only implemented for 2d situations")
			break
		print "Fourier transforming mode" + str(whichmode)
		dq=2.0*np.pi/self.conf.geom.Lx
		nq=int(qmax/dq)
		print "Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		fourierlong0=np.zeros((nq,nq),dtype=complex)
		fouriertrans0=np.zeros((nq,nq),dtype=complex)
		# #self.eigvec[0:2*self.N:2,u]
		eigx=self.eigvec[0:2*self.N:2,whichmode]
		eigy=self.eigvec[1:2*self.N:2,whichmode]
		for kx in range(nq):
			for ky in range(nq):
				# we need to be doing longitudinal and transverse here
				# Both have the same FT, but the local bits are q . e, and q X e
				fourierlong0[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.conf.rval[:,0]+qy[ky]*self.conf.rval[:,1]))*(eigx*qx[kx]+eigy*qy[ky])/np.sqrt(qx[kx]**2+qy[ky]**2+eps))/len(self.rval[:,0])
				fouriertrans0[kx,ky]=np.sum(np.exp(1j*(qx[kx]*self.conf.rval[:,0]+qy[ky]*self.conf.rval[:,1]))*(eigx*qy[ky]-eigy*qx[kx])/np.sqrt(qx[kx]**2+qy[ky]**2+eps))/len(self.rval[:,0])
		sqlong=np.real(fourierlong0**2+np.imag(fourierlong0)**2)
		sqtrans=np.real(fouriertrans0**2+np.imag(fouriertrans0)**2)
		# Produce a radial averaging to see if anything interesting happens
		nq2=int(2**0.5*nq)
		valrad=np.zeros((nq2,2))
		fourierlong=np.zeros((nq2,))
		fouriertrans=np.zeros((nq2,))
		for l in range(nq2):
			fourierlong[l]=np.mean(sqlong[ptsx[l],ptsy[l]])
			fouriertrans[l]=np.mean(sqtrans[ptsx[l],ptsy[l]])
		if verbose:
			plt.figure()
			plt.plot(qrad,fourierlong,'.-k')
			plt.plot(qrad,fouriertrans,'.-r')
			plt.xlabel('q')
			plt.ylabel('|xi_q|^2')
			plt.title('Mode ' + str(whichmode))
		return qrad,fourierlong,fouriertrans
		
	# project the modes into Fourier space to see how it's scaling
	def ModesFourier(self,whichmode,qmax=0.3,verbose=True,eps=0.001):
		# Strictly 2d affair
		if self.dim != 2:
			print("Error: Hessial - Fourier transform is only implemented for 2d situations")
			break
		print "Fourier transforming mode" + str(whichmode)
		dq=np.pi/self.conf.geom.Lx
		nq=int(qmax/dq)
		print "Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		fouriertrans=np.zeros((nq,nq,2),dtype=complex)
		eigx=self.eigvec[0:2*self.N:2,whichmode]
		eigy=self.eigvec[1:2*self.N:2,whichmode]
		for kx in range(nq):
			for ky in range(nq):
				# we need to be doing longitudinal and transverse here
				# Both have the same FT, but the local bits are q . e, and q X e
				fouriertrans[kx,ky,0]=np.sum(np.exp(1j*(qx[kx]*self.conf.rval[:,0]+qy[ky]*self.conf.rval[:,1]))*eigx)/self.N #len(self.rval[:,0])
				fouriertrans[kx,ky,1]=np.sum(np.exp(1j*(qx[kx]*self.conf.rval[:,0]+qy[ky]*self.conf.rval[:,1]))*eigy)/self.N # len(self.rval[:,0])
                Sq=np.real(fouriertrans[:,:,0])**2+np.imag(fouriertrans[:,:,0])**2+np.real(fouriertrans[:,:,1])**2+np.imag(fouriertrans[:,:,1])**2
		# Produce a radial averaging to see if anything interesting happens
		nq2=int(2**0.5*nq)
		Sqrad=np.zeros((nq2,))
		for l in range(nq2):
			Sqrad[l]=np.mean(Sq[ptsx[l],ptsy[l]])
		return qrad,Sqrad
	
	# get the elastic moduli cleanly once and for all. We won't diagonalise the dynamical matrix, but Fourier transform it instead
	# We have the dynamical matrix in real space as self.Hessian
	def getModuli(self,qmax=1.5,verbose=True):
		# Strictly 2d affair
		if self.dim != 2:
			print("Error: Hessian - gettin moduli is only implemented for 2d situations")
			break
		print "Fourier transforming Hessian"
		dq=2.0*np.pi/self.conf.geom.Lx
		nq=int(qmax/dq)
		print "Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps."
		qx, qy, qrad, ptsx, ptsy=self.makeQrad(dq,qmax,nq)
		print "After qrad"
		longitudinal0=np.zeros((nq,nq))
		transverse0=np.zeros((nq,nq))
		for k in range(nq):
			kx=qx[k]
			for l in range(nq):
				ky=qy[l]
				if verbose:
					print kx
					print ky
				# In Fourier space, for a given k (vector), we define the 2x2 k hessian as
				khessian=np.zeros((2,2),dtype=complex)
				khessian[0,0]=np.dot(np.exp(1j*kx*self.rval[:,0]),np.dot(self.Hessian[0:2*self.N:2,0:2*self.N:2],np.exp(-1j*kx*self.rval[:,0])))/self.N
				khessian[0,1]=np.dot(np.exp(1j*kx*self.rval[:,0]),np.dot(self.Hessian[0:2*self.N:2,1:2*self.N:2],np.exp(-1j*ky*self.rval[:,1])))/self.N
				khessian[1,0]=np.dot(np.exp(1j*ky*self.rval[:,1]),np.dot(self.Hessian[1:2*self.N:2,0:2*self.N:2],np.exp(-1j*kx*self.rval[:,0])))/self.N
				khessian[1,1]=np.dot(np.exp(1j*ky*self.rval[:,1]),np.dot(self.Hessian[1:2*self.N:2,1:2*self.N:2],np.exp(-1j*ky*self.rval[:,1])))/self.N
				# Its eigenvalues are then (B+\mu) k^2 and k^2, in principle
				eigk, eigveck = LA.eig(khessian)
				# We won't get proper, pure longitudinal and transverse eigenvectors
				# project them onto k, and take the one more along k as the longitudinal one
				proj1= kx*eigveck[0,0]+ky*eigveck[0,1]
				proj2= kx*eigveck[1,0]+ky*eigveck[1,1]
				if abs(proj2)>abs(proj1):
					longitudinal0[k,l]=eigk[1]
					transverse0[k,l]=eigk[0]
				else:
					longitudinal0[k,l]=eigk[0]
					transverse0[k,l]=eigk[1]
				if verbose:
					print "Found eigenvalues long " + str(longitudinal0[k,l]) + " and transverse " + str(transverse0[k,l])
		nq2=int(2**0.5*nq)
		longitudinal=np.zeros((nq2,))
		transverse=np.zeros((nq2,))
		for l in range(nq2):
			longitudinal[l]=np.mean(longitudinal0[ptsx[l],ptsy[l]])
			transverse[l]=np.mean(transverse0[ptsx[l],ptsy[l]])
		
		if verbose:
			plt.figure()
			plt.plot(qrad**2,longitudinal,'ok')
			plt.plot(qrad**2,transverse,'or')
		qrad2 = qrad**2
		return qrad2, longitudinal, transverse
		
