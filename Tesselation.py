

from Configuration import *
from CellList import *
import scipy.spatial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

MMAX=2.0
class Tesselation:
    
	def __init__(self,conf,debug=False):
		# First check for suitable data format of configuration
		if conf.multiopt == "many":
			print("Tesselation: Error - Can only construct a Tesselation with a single set of input coordinates. Construct single-position configuration instead.")
			sys.exit()
		self.conf=conf
		self.rval=self.conf.rval
		self.geom=self.conf.geom
		self.debug=debug
		self.ordered_patches = False
             
	# Create tesselation patches using the dual to a Delaunay triangulation
	# Use scip.spatial for a plane, and convex hull for a sphere (or related, not been tested)
	def findLoopDelaunay(self,debug=False):
		self.LoopList=[]
		# The dual: which loops belong to which particle
		self.ParList=[[] for k in range(len(self.rval))]
		self.LoopCen=[]
		self.l=0
		# use properly two-dimensional data
		# This needs to take into account the geometry now. Plane, periodic plane, and sphere are allowed
		# plane: simplest case
		if (self.geom.manifold =='plane'):
			# For non-periodic plane: works out of the box
			if (self.geom.periodic == False):
				tri = scipy.spatial.Delaunay(self.rval[:,0:2])
				loops=tri.simplices
				if debug:
					plt.figure()
					plt.triplot(self.rval[:,0], self.rval[:,1], tri.simplices.copy())
					plt.plot(self.rval[:,0], self.rval[:,1], 'o')
					plt.show()
			# for a periodic plane, decorate with half system-width copy all around 
			# This algorithm is vulnerable to empty patches with system-size triangles
			else:
				tri0 = scipy.spatial.Delaunay(self.rval[:,0:2])
				print("Initially " + str(len(tri0.simplices)) + " triangles found.")
				# shift and wrap so that the hull edges are in the centre
				rshift =  self.geom.ApplyPeriodic2d(self.rval+0.5*np.array([self.geom.Lx,self.geom.Ly,0.0]))
				trishift = scipy.spatial.Delaunay(rshift[:,0:2])
				# And then do it again shifted half a system size up in x and y for those edges to be in the centre
				rshift_x =  self.geom.ApplyPeriodic2d(self.rval+0.5*np.array([self.geom.Lx,0.0,0.0]))
				rshift_y =  self.geom.ApplyPeriodic2d(self.rval+0.5*np.array([0.0,self.geom.Ly,0.0]))
				trishift_x = scipy.spatial.Delaunay(rshift_x[:,0:2])
				trishift_y = scipy.spatial.Delaunay(rshift_y[:,0:2])
				# And now go through the things and choose the appropiate ones 
				# Careful, simplices are not in the same order here
				# First, find all the triangle indices that are not in the boundary region in the initial triangulation
				# Definition of the boundary region. This assumes our tesselation is non-crazy with no system-size triangles
				pad=0.25*(0.5*self.geom.Lx+0.5*self.geom.Ly)
				# decent are the central ones 
				decent=[]
				for k in range(len(tri0.simplices)):
					centre=np.sum(self.rval[tri0.simplices[k],:],axis=0)/3.0
					if (abs(centre[0])<(0.5*self.geom.Lx-pad)) and (abs(centre[1])<(0.5*self.geom.Ly-pad)):
						decent.append(k)
				# get the unique labels
				tokeep=list(set(decent))
				# Now in reverse, find all the ones that one might want to add from the other set
				# Those are the ones which now find themselves in the bang centre of the shifted tesselations
				better=[]
				# shifted in both directions
				for k in range(len(trishift.simplices)):
					# careful here ..
					centre=np.sum(rshift[trishift.simplices[k],:],axis=0)/3.0
					if abs(centre[0])<pad:
						if abs(centre[1])<(0.5*self.geom.Lx-pad):
							better.append(k)
					if abs(centre[1])<pad:
						if abs(centre[0])<(0.5*self.geom.Ly-pad):
							better.append(k)
				nuggetx=[]
				# Then there is a final bit of those guys which are in the original centre, but at the edges of the shifted
				# delaunay bit. Take them together and replace
				# This is a partial stitch. If we were to use the neighbour information, it's not going to be correct. But simply the simplex list, it's fine
				# SH: 18.08.20: Not entirely sure why this is necessary, in fact
				# shifted in x only
				for k in range(len(trishift_x.simplices)):
					centre=np.sum(rshift_x[trishift_x.simplices[k],:],axis=0)/3.0
					if (abs(centre[0])<pad) and (abs(centre[1])<pad):
						nuggetx.append(k)
				nuggety=[]
				# shifted in y only
				for k in range(len(trishift_y.simplices)):
					centre=np.sum(rshift_y[trishift_y.simplices[k],:],axis=0)/3.0
					if (abs(centre[0])<pad) and (abs(centre[1])<pad):
						nuggety.append(k)
				toadd=list(set(better))
				toaddx=list(set(nuggetx))
				toaddy=list(set(nuggety))
				
				loops=np.zeros((len(tokeep)+len(toadd)+len(toaddx)+len(toaddy),3),dtype=np.int)
				loops[:len(tokeep),:]=tri0.simplices[tokeep,:]
				loops[len(tokeep):(len(tokeep)+len(toadd)),:]=trishift.simplices[toadd,:]
				loops[(len(tokeep)+len(toadd)):(len(tokeep)+len(toadd)+len(toaddx)),:]=trishift_x.simplices[toaddx,:]
				loops[(len(tokeep)+len(toadd)+len(toaddx)):(len(tokeep)+len(toadd)+len(toaddx)+len(toaddy)),:]=trishift_y.simplices[toaddy,:]
				print("At the end we have " + str(len(loops)) + " triangles.")
				if debug:
					plt.figure()
					plt.triplot(self.rval[:,0], self.rval[:,1], tri0.simplices[tokeep],color='r')
					plt.triplot(self.rval[:,0], self.rval[:,1], trishift.simplices[toadd],color='g')
					plt.triplot(self.rval[:,0], self.rval[:,1], trishift_x.simplices[toaddx],color='b')
					plt.triplot(self.rval[:,0], self.rval[:,1], trishift_y.simplices[toaddy],color='m')
					#plt.triplot(self.rval[:,0], self.rval[:,1], loops)
					plt.show()
			# Delaunay - Voronoi algorithm by default gives me patches which are ordered clockwise,
			# not counterclockwise
			# tri.simplices are the triangles, but not necessarily counterclockwise ...
			# By default, they are clockwise. Therefore, reverse them
			for k in range(len(loops)):
					llist0=np.array(loops[k])
					llist=list(reversed(llist0))
					looppos=self.rval[llist]
					# also needs periodic BC: bring them all to the same side as first element
					looppos=looppos[0,:]+self.geom.ApplyPeriodic2d(looppos-looppos[0,:])
					lcen=[np.mean(looppos[:,0]), np.mean(looppos[:,1]),np.mean(looppos[:,2])]
					self.LoopCen.append(lcen)
					self.LoopList.append(llist)
					self.l+=1
			print("Found " + str(len(self.LoopList)) + " loops!")
		elif (self.geom.manifold == 'sphere'):
			# A little bird says that this is just the convex hull of my points
			print("Using convex hull to compute Delaunay triangulation on the sphere")
			tri = scipy.spatial.ConvexHull(self.rval)
			#print tri.simplices
			if debug:
				fig = plt.figure()
				ax = fig.add_subplot(1, 1, 1, projection='3d')
				ax.plot_trisurf(self.rval[:,0], self.rval[:,1], self.rval[:,2], triangles=tri.simplices)
				plt.show()
			loops=tri.simplices
			# However, these are randomly oriented, so we need to make them all counterclockwise
			for k in range(len(loops)):
				llist0=np.array(loops[k])
				# get the unit normal at the (geometric) centre of the triangle 
				rcent = np.mean(self.rval[llist0,:],axis=0)
				norm = self.geom.UnitNormal1d(rcent)
				# compute the signed area of the triangle
				r01=self.rval[llist0[1],:]-self.rval[llist0[0],:]
				r12=self.rval[llist0[2],:]-self.rval[llist0[1],:]
				arvec = np.cross(r12,r01)
				if (np.dot(arvec,norm) < 0.0):
					llist=list(reversed(llist0))
				else:
					llist=list(llist0)
				looppos=self.rval[llist]
				# also needs periodic BC: bring them all to the same side as first element
				looppos=looppos[0,:]+self.geom.ApplyPeriodic2d(looppos-looppos[0,:])
				lcen=[np.mean(looppos[:,0]), np.mean(looppos[:,1]),np.mean(looppos[:,2])]
				self.LoopCen.append(lcen)
				self.LoopList.append(llist)
				self.l+=1
			print("Found " + str(len(self.LoopList)) + " loops!")
		else:
			print("Error: Delaunay algorithm does not exist for geometry " + self.geom.manifold)
			print("Returning empty list. Use regular contact based tesselation instead.")
			return [],[],[]
			
		# Need to also construct the connectivity matrix
		# There surely are better ways ...
		self.Ival=[]
		self.Jval=[]
		for k in range(len(loops)):
			llist=loops[k]
			self.Ival.append(llist[0])
			self.Jval.append(llist[1])
			self.Ival.append(llist[1])
			self.Jval.append(llist[0])
			
			self.Ival.append(llist[1])
			self.Jval.append(llist[2])
			self.Ival.append(llist[2])
			self.Jval.append(llist[1])
			
			self.Ival.append(llist[0])
			self.Jval.append(llist[2])
			self.Ival.append(llist[2])
			self.Jval.append(llist[0])
			self.ParList[llist[0]].append(k)
			self.ParList[llist[1]].append(k)
			self.ParList[llist[2]].append(k)
		return self.LoopList,self.Ival,self.Jval
		
	# Construct tesselation based on contact networks
	# Option closeHoles does just that iteratively for systems with strong density fluctuations
	# Strongly preferred option for inhomogeneous systems
	# Only option for irregular surfaces
	def findLoop(self,closeHoles=False,zmin=3,mult0=1.0,mult1=MMAX):
		# We will need local coordinates on the surface
		self.conf.getTangentBundle()
		
		neighList=[]
		self.Ival=[]
		self.Jval=[]
		if closeHoles:
			Inei=[[] for k in range(len(self.rval))]
		else:
			Inei=[]
		count=0
		# Identify all neighbours and add them to a list. Keep i->j and j->i separate
		# The label is in neighList, the particle numbers are in Ival and Jval
		dmax=2*self.conf.sigma
		mult=mult0
		print("Max distance: "+ str(dmax))
		print("Initial multiplier " + str(mult0))
		for i in range(len(self.rval)):
			neighbours=[]
			if i%1000==0:
				print(i)
			if closeHoles:
				mult=mult0
				while len(neighbours)<zmin and mult<mult1:
					neighbours=self.conf.getNeighbours(i,mult,dmax)[0]
					mult=1.1*mult
					#print i, ' --> ', neighbours
				# find the new neighbours (because of our asymmetric contact algorithm there are neighbours which haven't been found
				# for a particle if it is far away from the others, and the others are in a high density region)
				jexisting=[]
				for n in Inei[i]:
					jexisting.append(self.Jval[n])
				neighs_new = [a for a in neighbours if a not in jexisting]
				neighList.extend([u for u in range(count,count+len(neighs_new))])
				self.Ival.extend([i for k in range(len(neighs_new))])
				self.Jval.extend(neighs_new)
				Inei[i].extend([u for u in range(count,count+len(neighs_new))])
				count+=len(neighs_new)
				# and the reverse contacts
				neighList.extend([u for u in range(count,count+len(neighs_new))])
				self.Jval.extend([i for k in range(len(neighs_new))])
				self.Ival.extend(neighs_new)
				# Need to find to which particles the other contacts of I (those <i) have been assigned
				for a in range(len(neighs_new)):
					Inei[neighs_new[a]].append(count+a)
				count+=len(neighs_new)
			else:
				neighbours=self.conf.getNeighbours(i,mult0,dmax)[0]
				#print len(neighbours)
				neighList.extend([u for u in range(count,count+len(neighbours))])
				self.Ival.extend([i for k in range(len(neighbours))])
				#self.Jval.extend(neighs[neighbours])
				self.Jval.extend(neighbours)
				Inei.append([u for u in range(count,count+len(neighbours))])
				count+=len(neighbours)
				#print neighbours
				if i%1000==0:
					print(neighbours)
		# Identify loops based on the neighbour list. Kick out any (one-way) contacts that have occured so far
		Jarray=np.array(self.Jval)
		self.LoopList=[]
		# The dual: which loops belong to which particle
		self.ParList=[[] for k in range(len(self.rval))]
		self.LoopCen=[]
		self.l=0
		print("Found " + str(len(neighList)) + " neighbours.")
		while len(neighList)>0:
			idx=neighList[0]
			idxkeep=idx
			idx0=[]
			llist=[]
			goneround=False
			while goneround==False:  
				# Sort neighbours counterclockwise according to their local angle  
				if self.conf.geom.periodic:
					dr0hat=self.geom.ApplyPeriodic11(self.rval[self.Ival[idx],:],self.rval[self.Jval[idx],:])
				else:
					dr0hat=self.rval[self.Jval[idx],:]-self.rval[self.Ival[idx],:]
				dr0hat/=np.sqrt(np.sum(dr0hat**2))
				jnei0=Inei[self.Jval[idx]]
				jnei=list(Jarray[jnei0])  
				if self.conf.geom.periodic:
					drvec=self.geom.ApplyPeriodic12(self.rval[self.Jval[idx],:],self.rval[jnei,:])
				else:
					drvec=self.rval[jnei,:]-self.rval[self.Jval[idx],:]
				drhat=((drvec).transpose()/(np.sqrt(np.sum(drvec**2,axis=1))).transpose()).transpose()
				cbeta=np.einsum('kj,j->k',drhat,self.conf.e2[self.Jval[idx],:])
				sbeta=np.einsum('kj,j->k',drhat,self.conf.e1[self.Jval[idx],:])
				cbeta0=np.dot(dr0hat,self.conf.e2[self.Jval[idx],:])
				sbeta0=np.dot(dr0hat,self.conf.e1[self.Jval[idx],:])
			
				# arccos returns between 0 and pi. Just multiply by the sign of the sine
				beta=np.arccos(cbeta)*np.sign(sbeta)
				# Determine the angles from the contact (read backwards) to the others, and pick the largest, modulo 2pi
				beta0=np.arccos(cbeta0)*np.sign(sbeta0)-np.pi
				dbeta=beta-beta0
				dbeta-=2*np.pi*np.round((dbeta-np.pi)/(2*np.pi))
				# and throwing out the particle itself
				itself=jnei.index(self.Ival[idx])
				dbeta[itself]=-1
				cnt=np.argmax(dbeta)
			
				idx=jnei0[cnt]
				goneround = idx in idx0
				if goneround==False:
					idx0.append(idx)
					llist.append(Jarray[idx])
					self.ParList[Jarray[idx]].append(self.l)
			for v in idx0:
				#print v
				#neighList.remove(v)
				try:
					neighList.remove(v)
				except ValueError:
					pass
			# There may be rare isolated cases (rattlers?) where the first contact itself is not part of the eventual loop.
			# This causes problems, because the loop identified after that has been removed.
			# Remove the original contact, in case it hasn't
			#neighList.remove(idxkeep)
			try:
				#print idxkeep
				neighList.remove(idxkeep)
			except ValueError:
				pass
			looppos=self.rval[llist]
			# Apply periodic boundary conditions here too ...
			# Arbitrary decision - go with the side of the first one
			if self.geom.periodic:
				dl=self.geom.ApplyPeriodic12(looppos[0,:],looppos)
				looppos=looppos[0,:]+dl
			lcen=[np.mean(looppos[:,0]), np.mean(looppos[:,1]),np.mean(looppos[:,2])]
			# This centre can be in a dent for curved surfaces
			# Easy to correct for a sphere
			# Other geometries are not implemented so far
			if self.conf.geom.manifold=='sphere':
				llen=np.sqrt(lcen[0]**2+lcen[1]**2+lcen[2]**2)
				lcen/=llen
				lcen*=self.conf.geom.R
			self.LoopCen.append(lcen)
			self.LoopList.append(llist)
			self.l+=1
		print("Found " + str(len(self.LoopList)) + " loops!")
		return self.LoopList,self.Ival,self.Jval
      
	# Much prettier: a loop that is too big (as measured by the mean square distance of the distances to the particles)
	# Deconstruct it into lots of little loops (virtual ones), with defined centers
	def makeEdges(self,rmax,maxlen=20):
		for l0 in range(len(self.LoopList)):
			llist=self.LoopList[l0]
			looppos=self.rval[llist]
			if self.geom.periodic:
				dlvec=self.geom.ApplyPeriodic12(self.LoopCen[l0],looppos)
			else:
				dlvec=looppos-self.LoopCen[l0]
			isLong=np.sqrt(np.sum(np.sum(dlvec**2,axis=1)))/len(llist)
			if len(llist)>maxlen:
				print("Loop " + str(l0) + " with particles " + str(llist) + " is too big! ")
				for k in range(len(llist)):
					kside=k-1
					if kside<0:
						kside=len(llist)-1
					# Attempting to catch the inward pointing loops: the have to be global boundary ~sqrt(N)
					if len(llist)<0.5*np.sqrt(len(self.rval)):
						newcen=0.5*(self.rval[llist[k]]+self.rval[llist[kside]])-self.conf.inter.sigma*dlvec[k,:]/np.sqrt(np.sum(dlvec[k,:]**2))
					else:
						newcen=0.5*(self.rval[llist[k]]+self.rval[llist[kside]])+self.conf.inter.sigma*dlvec[k,:]/np.sqrt(np.sum(dlvec[k,:]**2))
					self.LoopCen.append(newcen)
					try:
						self.ParList[llist[k]].remove(l0)
					except ValueError:
						pass
					self.ParList[llist[k]].append(self.l)
					try:
						self.ParList[llist[kside]].remove(l0)
					except ValueError:
						pass
					self.ParList[llist[kside]].append(self.l)
					self.l+=1
        
	# While we are at it, we can construct the dual tesselation here.
	# All that's missing is to order the patches for the particles counterclockwise
	def OrderPatches(self):
		LoopCen1=np.array(self.LoopCen)
		for i in range(len(self.rval)):
			parray=np.array(self.ParList[i])
			#print parray
			if self.geom.periodic:
				drvec=self.geom.ApplyPeriodic12(self.rval[i,:],LoopCen1[self.ParList[i]])
			else:
				drvec=LoopCen1[self.ParList[i]]-self.rval[i,:]
			# Optionally Take care of irregularities (in the form of too long bonds) here. These happen at the edges of connected stuff
			# The tesselation is correct, it's just not what we want
			drlen=np.sqrt(np.sum(drvec**2,axis=1))
			drhat=((drvec).transpose()/(drlen).transpose()).transpose()
			cbeta=np.einsum('kj,j->k',drhat,self.conf.e1[i,:])
			sbeta=np.einsum('kj,j->k',drhat,self.conf.e2[i,:])
			# arccos returns between 0 and pi. Just multiply by the sign of the sine
			beta=np.arccos(cbeta)*np.sign(sbeta)
			# sort by angle and put back in ParList
			lorder=np.argsort(beta)
			self.ParList[i]=parray[lorder] 
			# Use the new ParList structure where loops belong to particles are stored
		self.ordered_patches = True
		return self.LoopList,self.LoopCen,self.ParList,self.Ival,self.Jval

	# Tile areas
	def ComputePatchArea(self):
		if not self.ordered_patches: 
			raise Exception('Patches have to be ordered in order to cumpute their area.')
		self.area = []
		for k in xrange(len(self.ParList)):
			if len(self.ParList[k])>0:
				xc, yc, zc = 0.0, 0.0, 0.0
				for l in self.ParList[k]:
					xc += self.LoopCen[l][0]
					yc += self.LoopCen[l][1]
					zc += self.LoopCen[l][2]
				xc /= len(self.ParList[k])
				yc /= len(self.ParList[k])
				zc /= len(self.ParList[k])
				N = len(self.ParList[k])
				area = 0.0
				for i in xrange(N):
					#print self.ParList[i]
					x1 = self.LoopCen[self.ParList[k][i]][0] - xc
					y1 = self.LoopCen[self.ParList[k][i]][1] - yc
					z1 = self.LoopCen[self.ParList[k][i]][2] - zc
					x2 = self.LoopCen[self.ParList[k][(i+1)%N]][0] - xc
					y2 = self.LoopCen[self.ParList[k][(i+1)%N]][1] - yc 
					z2 = self.LoopCen[self.ParList[k][(i+1)%N]][2] - zc
					n = np.cross([x1,y1,z1],[x2,y2,z2])
					area += 0.5*np.sqrt(np.dot(n,n))
			else:
				area=0.0
			self.area.append(area)
			
	#  virtual contractile stress. Not relevant anymore, keeping for now		
	def computeContractile(self,alpha):
		self.ComputePatchArea()
		N = len(self.rval)
		stress = np.zeros((N,3,3))
		pressure = np.zeros((N,))
		for i in range(N):
			# simple contractile part n x n
			#stress[i,:,:]=alpha*np.einsum('ij,kl->self.nval,self.nval)
			pressure[i]=alpha*self.conf.param.pot_params['k']*np.sum(self.conf.nval[i,:]**2)*self.area[i]
		return pressure

