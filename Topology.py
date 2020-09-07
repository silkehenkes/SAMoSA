from Configuration import *
from Tesselation import *
from Defects import *
from numpy import linalg as LA

try:
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.cm as cm
	HAS_MATPLOTLIB = True
except:
	HAS_MATPLOTLIB = False
	pass

# Warning: Implicitly spherical geometry. Results in other geometries will be garbage (should not crash though)
# This is a child class of Configuration and handles all defect and profile analysis
# We are assuming that we are already using the set of particles that we want. Construct with proper usetype, e.g. usetype = [1,2] for cornea.
class Topology(Configuration):
	
	# Use the Configuration default constructor. We *need* to be set to self.multiopt == "many"!
	def validate_initialise():
		if not self.multiopt == "many":
			print("Topology:: Error: Cannot run full topology without reading in multiple files! Use manual Configuration / Tesselation / Defect for a single file instead.")
			sys.exit()
		# Some booleans to avoid redoing the whole shebang twice as some computations depend on others
		hasWriter = False
		# Is there a current child configuration?
		hasChild = False
		
	def getFlowField(self,inisnap,dsnap=1,debug=False):
		# attempt to compute the flow field between two snapshots, based on the uniquely labeled particles present in both
		flag1=list(self.flag[inisnap,:self.Nval[inisnap]])
		flag2=list(self.flag[inisnap+dsnap,:self.Nval[inisnap+dsnap]])
		#print(flag1)
		#print(flag2)
		# intersection: do this manually, possibly easier ...
		# labels in order for both
		useparts1=[]
		useparts2=[]
		index=[]
		hasdied=[]
		for k1 in range(len(flag1)):
			try:
				k2=flag2.index(flag1[k1])
				useparts1.append(k1)
				useparts2.append(k2)
				index.append(flag1[k1])
			except:
				if debug:
					print("particle " + str(flag1[k1]) + " died.")
				hasdied.append(k1)
		#print(index)
		
		# Make this the actual displacement field
		flowField = (self.rval[inisnap+dsnap,useparts2,:]-self.rval[inisnap,useparts1,:])
		
		# also generate a polarisation field that is averaged over the snaps in between
		nsnap = dsnap+1
		polarField = self.nval[inisnap,useparts1,:]/nsnap
		for u in range(1,nsnap):
			print(u)
			flag2=list(self.flag[inisnap+u,:self.Nval[inisnap+u]])
			#print(flag2)
			useparts2=[]
			for k in range(len(useparts1)):
				k2 = flag2.index(index[k])
				useparts2.append(k2)
			polarField = polarField +  self.nval[inisnap+u,useparts2,:]/nsnap
		# need to normalise again for tracking
		nnorm = np.sqrt(polarField[:,0]**2 + polarField[:,1]**2+polarField[:,2]**2)
		polarField = polarField / np.outer(nnorm,np.ones((3,)))
			
		
		if debug:
			plt.figure(figsize=(8,8))
			velangle = np.arctan2(flowField[:,1],flowField[:,0])
			#arctan2 gives results between -pi to pi, normalise ...
			colors = (velangle+np.pi)/(2*np.pi)
			colormap = cm.hsv
			plt.quiver(self.rval[inisnap,useparts1,0],self.rval[inisnap,useparts1,1],flowField[:,0],flowField[:,1],color=0.8*colormap(colors))
			plt.gca().set_aspect('equal')
			plt.title('Skipping ' + str(dsnap))
			
			plt.figure(figsize=(8,8))
			velangle = np.arctan2(self.vval[inisnap,:,1],self.vval[inisnap,:,0])
			#arctan2 gives results between -pi to pi, normalise ...
			colors = (velangle+np.pi)/(2*np.pi)
			colormap = cm.hsv
			plt.quiver(self.rval[inisnap,useparts1,0],self.rval[inisnap,useparts1,1],self.vval[inisnap,:,0],self.vval[inisnap,:,1],color=0.8*colormap(colors))
			plt.gca().set_aspect('equal')
			plt.title('Instantaneous velocity field')
			
		return useparts1, flowField, polarField
	
	def makeFlowChild(self,frame,useparts,FlowField,PolarField):
		# To handle this, generate a child configuration with the flow field as velocities.
		# FIX: Copy parameters, currently. Go down a level of complexity later
		param0 = self.param
		# Take out the persistent particles of frame frame for which the flow field is defined
		rval0 = self.rval[frame,useparts,:]
		radius0 = self.radius[frame,useparts]
		ptype0 = self.ptype[frame,useparts]
		flag0 = self.flag[frame,useparts]
		
		# Generate child configuration (not through makeChild because we use flowField as velocities)
		flowChild = Configuration(initype="fromPython",param=param0,rval=rval0,vval=FlowField,nval=PolarField,radii=radius0,ptype=ptype0,flag=flag0)
		return flowChild
		
	def makeFrameChild(self,frame):
		# first generate the appropriate child configuration for only that frame
		frameChild = Configuration(initype="makeChild",frame=frame,usetype="all")
		return frameChild
	
	
	# generic getDefects algorithm
	# use with frameChild to track defects in a particular frame
	# use with flowChild and field = "velocity" and symtype = "polar" to track the flow field on the cornea
	# Track the (polar) defects on a corneal flow field
	def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False):
		
		# Now generate tesselation and defects
		tess = Tesselation(child)
		print("initialized tesselation")
		if delaunay:
			LoopList,Ival,Jval = tess.findLoopDelaunay()
		else:
			#findLoop(self,closeHoles=False,zmin=3,mult0=1.0,mult1=MMAX):
			LoopList,Ival,Jval = tess.findLoop(closeHoles,zmin,mult)
		print("found loops")
		df = Defects(tess,child)
		defects0,numdefect0=df.getDefects(symtype,field)
		# Clean up and merge the resulting defects
		defects,numdefect = df.mergeDefects(defects0,numdefect0,rmerge)
		#defects = defects0
		#numdefect = numdefect0
		print("After merging field " + field + " with symtype " + symtype + " and mrege radius " + str(rmerge) + " found " + str(numdefect) + " defects:")
		print(defects)
		
		# tesselation for writer ... less than elegant
		return defects, numdefect, tess
	
	# locate the central defect. Starting point: pure geometry
	def centralDefect(self,child,defects,numdefect,maxangle=0.3*np.pi):
		# translate that maxangle into max flat x y distance:
		maxdist = child.geom.R*np.sin(maxangle)
		if numdefect ==1:
			return np.array(defects[0][1:4])
		else:
			options = []
			for k in range(numdefect):
				dist = np.sqrt(defects[k][1]**2+defects[k][2]**2)
				if dist < maxdist:
					print("Potential central defect!")
					options.append(k)
			print(options)
			if len(options)==0:
				print("No central defect ...")
				return "problem"
			elif len(options)==1:
				return np.array(defects[options[0]][1:4])
			elif len(options)==2:
				# Merge them
				pos = 0.5*(np.array(defects[options[0]][1:4])+np.array(defects[options[1]][1:4]))
				return pos
			else:
				print("A horrible mess")
				return "problem"
					
		

	# On the sphere with bands: locate the orientation of the flow of material on there
	# Reorient the data set
	# Again, will use a child configuration. 
	# For cornea flow field, use with flowChild and nematic = False. Note that this is based on the flow then, not the defect ...
	def redressTiltGeometric(self,child,nematic=False,debug=False):
		# This only makes sense on a sphere
		if not child.geom.manifold =='sphere':
			print("Topology: Error - attempting to reorient a non-spherical configuration. This makes no sense. Stopping.")
			sys.exit()
	
		#self.rval=self.conf.rval
		ez = np.array([0,0,1])  # lab frame z-axis
		# Simply get the axis as the mean crossproduct or r and v; assuming alignment. This should also not flip.
		if not nematic:
			direction=np.sum(np.cross(child.rval,child.vval),axis=0)
		# Otherwise we can't do this since it's going to be close to 0
		else:
			print("doing nematic case!")
			# Be a bit better here. The moment of inertia tensor should have *some* kind of signature
			# Take this as an initial guess of the axis. Compare the two later.
			Itensor=np.einsum('ik,ij->kj',child.rval,child.rval)
			# Compute its eigenvalues and eigenvectors
			eigval,eigvec=LA.eig(Itensor)
			# The eigenvector we want is the one with the *smallest* eigenvalue, as our shape
			# is a squashed thing with the z axis in the squattest direction
			idx=np.argmin(eigval)
			inertialz=eigvec[:,idx]
			print(inertialz)
			
			directions=np.cross(child.rval,child.vval)
			# Those should then now be mostly either aligned or antialigned
			# Hope for the best and rectify them ...
			# Use the guess from the inertia tensor now
			normdir=np.sqrt(directions[:,0]**2+directions[:,1]**2+directions[:,2]**2)
			dirnorm=((directions).transpose()/(normdir).transpose()).transpose()
			orient=np.round(dirnorm[:,0]*inertialz[0]+dirnorm[:,1]*inertialz[1]+dirnorm[:,2]*inertialz[2])
			direction=np.einsum('ij,i->j',directions,orient)
		print(direction)
		orderpar=direction/child.N
		print(orderpar)
		#self.direction = self.direction/np.linalg.norm(self.direction)
		# FIX: Overwriting the other estimate here
		direction=inertialz
		
		# This will be the axis angle around which we need to rotate the configuation for the director to point along z
		axis = np.cross(direction,ez)
		axis = axis/np.linalg.norm(axis)
		rot_angle = np.arccos(np.dot(direction,ez))
		print(rot_angle)
		
		if debug:
			# Debugging output
			if HAS_MATPLOTLIB:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(child.rval[:,0], child.rval[:,1], child.rval[:,2], zdir='z', c='b')
				ax.plot([-child.geom.R*direction[0],child.geom.R*direction[0]],[-child.geom.R*direction[1],child.geom.R*direction[1]],[-child.geom.R*direction[2],child.geom.R*direction[2]],'o-r')
				ax.plot([0,0],[0,0],[-child.geom.R,child.geom.R],'o-g')
				ax.plot([-child.geom.R*axis[0],child.geom.R*axis[0]],[-child.geom.R*axis[1],child.geom.R*axis[1]],[-child.geom.R*axis[2],child.geom.R*axis[2]],'o-k')
			else:
				print('Error: Matplotlib does not exist on this machine, cannot plot system')
				
		axis0 = np.empty((childConf.N,3))
		axis0[:,0] = axis[0]
		axis0[:,1] = axis[1]
		axis0[:,2] = axis[2]
		
		# Now: actually rotate the configuration and redo the cell list
		child.rotateFrame(axis0,rot_angle)
		# Need to redo the cell list after the rotation
		child.redoCellList()
		
		return child, direction, orderpar, axis, rot_angle
	
	# Reorient configuation based on the position of a defect (as a simple coordinate triplet)
	# For Cornea, but potentially for other things too 
	def redressTiltDefect(self,child,defectpos,debug=False):
		
		# The direction of our symmetry axis is the defect position now
		direction=defectpos/np.linalg.norm(defectpos)
		ez = np.array([0,0,1])
	
		# This will be the axis angle around which we need to rotate the configuation for the director to point along z
		axis = np.cross(direction,ez)
		axis = axis/np.linalg.norm(axis)
		rot_angle = np.arccos(np.dot(direction,ez))
		print(rot_angle)
		
		if debug:
			# Debugging output
			if HAS_MATPLOTLIB:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(child.rval[:,0], child.rval[:,1], child.rval[:,2], zdir='z', c='b')
				ax.plot([-child.geom.R*direction[0],child.geom.R*direction[0]],[-child.geom.R*direction[1],child.geom.R*direction[1]],[-child.geom.R*direction[2],child.geom.R*direction[2]],'o-r')
				ax.plot([0,0],[0,0],[-child.geom.R,child.geom.R],'o-g')
				ax.plot([-child.geom.R*axis[0],child.geom.R*axis[0]],[-child.geom.R*axis[1],child.geom.R*axis[1]],[-child.geom.R*axis[2],child.geom.R*axis[2]],'o-k')
			else:
				print('Error: Matplotlib does not exist on this machine, cannot plot system')
				
		axis0 = np.empty((child.N,3))
		axis0[:,0] = axis[0]
		axis0[:,1] = axis[1]
		axis0[:,2] = axis[2]
		
		# Now: actually rotate the configuration and redo the cell list
		child.rotateFrame(axis0,rot_angle)
		
		return child, axis, rot_angle
	
	# Cornea profiles: Simpler, to be able to compare to the experiments
	# Directly adapted from corresponding code
	# There does appear to be an errant minus sign in the experimental etheta? 
	def getSwirlInward(self,child,field,thetamax =70/360.0*2*np.pi,nbin=50,verbose=False):
	
		swirl=np.zeros((child.N,))
		inward=np.zeros((child.N,))
		#thetaD=np.zeros((child.N,))
		
		# full tangent getTangentBundle
		theta,phi,etheta,ephi = child.getTangentBundle()
		for k in range(child.N):
			#thetaD[k], etheta, ephi = get_local_defect(x[k],y[k],z[k],zdir,radius)
			#swirl[k]=flow_field[0,k]*ephi[0]+flow_field[1,k]*ephi[1]+flow_field[2,k]*ephi[2]
			#inward[k]=flow_field[0,k]*etheta[0]+flow_field[1,k]*etheta[1]+flow_field[2,k]*etheta[2]
			if field == "velocity":
				swirl[k] = child.vhat[k,0]*ephi[k,0]+child.vhat[k,1]*ephi[k,1]+child.vhat[k,2]*ephi[k,2]
				inward[k] = -(child.vhat[k,0]*etheta[k,0]+child.vhat[k,1]*etheta[k,1]+child.vhat[k,2]*etheta[k,2])
			elif field == "orientation":
				swirl[k] = child.nval[k,0]*ephi[k,0]+child.nval[k,1]*ephi[k,1]+child.nval[k,2]*ephi[k,2]
				inward[k] = -(child.nval[k,0]*etheta[k,0]+child.nval[k,1]*etheta[k,1]+child.nval[k,2]*etheta[k,2])
			
		#plt.figure()
		#plt.plot(theta,swirl,'.r')
		#plt.plot(theta,inward,'.g')
			
		# and more cleanly binned
		
		thetabin=np.linspace(0,thetamax,nbin)
		dtheta=thetabin[1]-thetabin[0]
		# bins of individual ones
		binval = (np.floor(theta/dtheta)).astype(int)
		swirlhist=np.zeros((nbin,))
		swirlerr=np.zeros((nbin,))
		inhist=np.zeros((nbin,))
		inerr=np.zeros((nbin,))
		isdata=[]
		for b in range(nbin):
			inbin=np.where(binval==b)
			if len(inbin)>0:
				isdata.append(b)
				swirlhist[b]=np.mean(swirl[inbin])
				swirlerr[b]=np.std(swirl[inbin])/np.sqrt(len(inbin))
				inhist[b]=np.mean(inward[inbin])
				inerr[b]=np.std(inward[inbin])/np.sqrt(len(inbin))
		
		if verbose:
			plt.figure()
			plt.errorbar(thetabin[isdata]+dtheta/2,swirlhist[isdata],swirlerr[isdata],marker='o',color='r',label='swirl')
			plt.errorbar(thetabin[isdata]+dtheta/2,inhist[isdata],inerr[isdata],marker='o',color='g',label='inward')
			plt.xlabel('theta')
			plt.ylabel('flow component ' + str(field))
			plt.legend()
			
		return thetabin, isdata, swirlhist, inhist, swirlerr, inerr
		
	
	# compute the profiles of everything on a sphere
	# TO DO: Either adapt or second version to work on cornea swirl and inward measures
	def getProfiles(self,child,nbin,debug=False):
		
		eng, press,ncon,stress = child.compute_energy_and_pressure()
		
		vel = np.sqrt(child.vval[:,0]**2 + child.vval[:,1]**2 + child.vval[:,2]**2)
		#velnorm=((self.conf.vval).transpose()/(vel).transpose()).transpose()
  
		# Generate the local coordinate system
		child.getTangentBundle()
		
		# Project the stresses into the e,theta,phi components. The rr component hast to be 0, and the r cross components
		# belong to the projection. So they are not all that interesting. 
		# We want the theta theta, theta phi, phi theta ant phi phi components (implicitly testing symmetries ...)
		# I give up on the notation. Stress is (N,3,3), the axes are (N,3). We want e_i sigma_ij e_j
		s_tt=np.sum(child.etheta*np.einsum('kij,kj->ki',stress,child.etheta),axis=1)
		s_tp=np.sum(child.etheta*np.einsum('...ij,...j->...i',stress,child.ephi),axis=1)
		s_pt=np.sum(child.ephi*np.einsum('...ij,...j->...i',stress,child.etheta),axis=1)
		s_pp=np.sum(child.ephi*np.einsum('...ij,...j->...i',stress,child.ephi),axis=1)
		
		# Alpha, the angle between the local polarity and the equator; here represented by ephi
		alpha=-np.arcsin(np.sum(child.nval*child.etheta, axis=1))
		
		# Setting up the binning. I changed this to go from -pi/2 to pi/2 consistently. This maybe makes less pretty pictures,
		# but the edges are going to be a lot cleaner. Also only one bin to handle accross multiple v0/J.
		# Can always rebin to less resolution if necessary
		# Position angle with the z axis
		theta_bin=np.linspace(0,np.pi,nbin+1)
		dtheta=theta_bin[1]-theta_bin[0]
		theta_out=theta_bin[:nbin]+dtheta/2-np.pi/2
		
		rho_profile, bin_edges = np.histogram(child.theta, bins=theta_bin,density=True)
		isdata=[index for index,value in enumerate(rho_profile) if (value >0)]
		normz=2*np.pi*child.geom.R*abs(np.cos(theta_out))
		rho_profile[isdata]=rho_profile[isdata]/normz[isdata]
		rho_profile/=np.mean(rho_profile)
		vel_profile=np.zeros(np.shape(rho_profile))
		eng_profile=np.zeros(np.shape(rho_profile))
		press_profile=np.zeros(np.shape(rho_profile))
		s_tt_profile=np.zeros(np.shape(rho_profile))
		s_tp_profile=np.zeros(np.shape(rho_profile))
		s_pt_profile=np.zeros(np.shape(rho_profile))
		s_pp_profile=np.zeros(np.shape(rho_profile))
		alpha_profile=np.zeros(np.shape(rho_profile))
		for idx in range(nbin):
			inbin=[index for index,value in enumerate(child.theta) if (value >= theta_bin[idx]  and value<=theta_bin[idx+1])]
			#print len(inbin)
			if len(inbin)>0:
				vel_profile[idx]=np.mean(vel[inbin])
				eng_profile[idx]=np.mean(eng[inbin])
				press_profile[idx]=np.mean(press[inbin])
				s_tt_profile[idx]=np.mean(s_tt[inbin])
				s_tp_profile[idx]=np.mean(s_tp[inbin])
				s_pt_profile[idx]=np.mean(s_pt[inbin])
				s_pp_profile[idx]=np.mean(s_pp[inbin])
				alpha_profile[idx]=np.mean(alpha[inbin])
		
		# Debugging output
		if debug==True:
			if HAS_MATPLOTLIB:
				fig = plt.figure()
				ax = fig.add_subplot(111, projection='3d')
				ax.scatter(child.rval[:,0], child.rval[:,1], child.rval[:,2], zdir='z', c='b')
			else:
				print('Error: Matplotlib does not exist on this machine, cannot plot system')
			
		return [theta_out,rho_profile,vel_profile,eng_profile,press_profile,s_tt_profile,s_tp_profile,s_pt_profile,s_pp_profile,alpha_profile]
        
