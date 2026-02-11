
import sys
import argparse
import pickle

sys.path.insert(1,'/home/sh18581/Documents/SAMoSA/')

from Topology import *
from Writer import *
from read_param import *

#from Writer import *

# WTF is wrong with my plotting??
#import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conffile", type=str, default="cornea_large_260819.conf", help="configuration file")
parser.add_argument("-d", "--directory", type=str, default="/media/sh18581/Elements/CorneaSimulations/Kaja/CorneaSimulations/cornea181019_cone/radius_300/J_0.1/v0_3/nu_1.0/rdivide_0.2/rl_0.4/ratio_0.1/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=0, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=50, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=1, help="step snapshots with this spacing in flow field")
parser.add_argument("-a", "--average", type=int, default=1, help="average over these many samples in flow field")
parser.add_argument("--coneangle", type=int, default=30, help="cone angle in degrees")
parser.add_argument("--nodefects", action='store_true', default=False, help="Turn off defect tracking (for known disordered systems)")

args = parser.parse_args()

# Use the Configuration constructor in its readCSV format to generate the topology 
# fromCSV(kwargs["parampath"],kwargs["datapath"],kwargs["multiopt"])
parampath0 = args.directory + args.conffile
param = Param(parampath0)
Cornea = Topology(initype="fromCSV",param=param,datapath=args.directory,multiopt="many")
# Now read in as desired
# def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
Cornea.readDataMany("SAMoS",args.skip,args.step,args.howmany,True,readtypes = [1,2])
#def __init__(self,directory,conffile,skip,howmany,ignore=True,maxtype=3):

data={'configuration':args.conffile}

write = Writer()
output= True

nav = 0
nbin = 100

data['nbin']=nbin
data['directory']=args.directory
data['skip']=args.skip
data['howmany']=args.howmany
data['step']=args.step
data['average']=args.average
data['coneangle']=args.coneangle

# Histogram of swirling
v_swirlhist = np.zeros((args.howmany-args.average,nbin))
n_swirlhist = np.zeros((args.howmany-args.average,nbin))
v_swirlerr = np.zeros((args.howmany-args.average,nbin))
n_swirlerr = np.zeros((args.howmany-args.average,nbin))
v_inhist = np.zeros((args.howmany-args.average,nbin))
n_inhist = np.zeros((args.howmany-args.average,nbin))
v_inerr = np.zeros((args.howmany-args.average,nbin))
n_inerr = np.zeros((args.howmany-args.average,nbin))

v_isdata = np.zeros((args.howmany-args.average,nbin))
n_isdata = np.zeros((args.howmany-args.average,nbin))

nbin_flux = int(args.coneangle/360.0*2*np.pi*Cornea.geom.R/(2*Cornea.sigma))
isdata = np.zeros((args.howmany-args.average,nbin_flux))
velflux = np.zeros((args.howmany-args.average,nbin_flux))
velav = np.zeros((args.howmany-args.average,nbin_flux))
velproj = np.zeros((args.howmany-args.average,nbin_flux))
alpha = np.zeros((args.howmany-args.average,nbin_flux))
vel2av = np.zeros((args.howmany-args.average,nbin_flux))
birthcount = np.zeros((args.howmany-args.average,nbin_flux))
deathcount = np.zeros((args.howmany-args.average,nbin_flux))


# Defects
vdefout= []
nvdefout = []
ndefout= []
nndefout = []

# Tilting success
goodconf_n = []
goodconf_v = []
defuse_n = []
defuse_v = []

# Central defect positon
n_defpos = []
v_defpos = []

# Be a bit more conservative for the large one
# Start reducing this for smaller corneas
# R=400: stay at 20
# R=300: reduce to 15
rmerge=15
#rmerge=0
nuke=True
maxedge=50

data['rmerge']=rmerge
data['tesstype']='delaunay'
data['nuke']=nuke
data['maxedge']=maxedge

# Attempt to compute things now
# Exclude the boundary by stopping a couple of cells short from the cone angle. Cell diameter = 10 (micron)
# 3 cells in radians
# Be less conservative for the edge wound as the defect could be there
# Reduce for smaller corneas, make it proportionally 20% away from the edge
# R=400: down to 80
# R=300: down to 60
dboundary = 80.0/Cornea.geom.R
print(dboundary)
maxangle = args.coneangle/360.0*2*np.pi - dboundary
print('Going out to maximum angle ' + str(maxangle) + ' in degrees ' + str(maxangle/(2*np.pi)*360))

data['maxangle']=maxangle
	
thetabin=np.linspace(0,args.coneangle,nbin)	

nav = 0
#for k in range(1):
for k in range(args.howmany-args.average):
	birthflag, deathflag, birthpos, deathpos = Cornea.getBirthDeath(k,args.average)
	useparts, FlowField, PolarField = Cornea.getFlowField(k,args.average)
	flowChild = Cornea.makeFlowChild(k,useparts,FlowField,PolarField,False)
	# def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False):
	# getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False,nuke=True,maxedge=25, coneangle=70.0/360*2*np.pi):
	if not args.nodefects:
		defects_v, numdefect_v,tess = Cornea.getDefects(flowChild,'velocity','polar',rmerge,3,0.5,False,True,nuke=True,maxedge=maxedge,coneangle = args.coneangle/360.0*2*np.pi)
		defects_n, numdefect_n,tess = Cornea.getDefects(flowChild,'orientation','polar',rmerge,3,0.5,False,True,nuke=True,maxedge=maxedge,coneangle = args.coneangle/360.0*2*np.pi)
		
		vdefout.append(defects_v)
		nvdefout.append(numdefect_v)
		ndefout.append(defects_n)
		nndefout.append(numdefect_n)
	
	if output:
		tess.OrderPatches()
		write.writeConfigurationVTK(flowChild,'test_' +str(k) +'.vtp')
		#write.writePatches(tess,'test_patches_'+str(k)+'.vtp')
		if not args.nodefects:
			write.writeDefects(defects_v, numdefect_v,'test_flowdefects_' + str(k) + '.vtp')
			write.writeDefects(defects_n, numdefect_n,'test_polarisationdefects' +str(k) + '.vtp')
	
	
	if not args.nodefects:
		# Use the director defect to redress tilt for n field
		#def centralDefect(self,child,defects,numdefect,maxangle=0.3*np.pi):
		defectpos = Cornea.centralDefect(flowChild,defects_n,numdefect_n,maxangle=maxangle)
		n_defpos.append(defectpos)
		# Same thing but for v
		defectpos2 = Cornea.centralDefect(flowChild,defects_v,numdefect_v,maxangle=maxangle)
		v_defpos.append(defectpos2)
	else:
		defectpos = "problem"
		defectpos2 = "problem"
	
	# Checking if we have at least one type of profile.
	anyflux = False
	
	if defectpos == "problem":
		if not defectpos2 == "problem":
			defectpos = defectpos2
			print("Using velocity central defect for n field!")
			defuse_n.append('v')
	else:
		defuse_n.append('n')
	if defectpos2 == "problem":
		if not defectpos == "problem":
			defectpos2 = defectpos
			print("Using director central defect for v field!")
			defuse_v.append('n')
	else:
		defuse_v.append('v')
	if not defectpos == "problem":
		goodconf_n.append(k)
		# def redressTiltDefect(self,child,defectpos,births='none',deaths='none',debug=False):
		axis, rot_angle, birthpos, deathpos = Cornea.redressTiltDefect(flowChild,defectpos,birthpos,deathpos)
		#thetabin, isdata, swirlhist, inhist, swirlerr, inerr = getSwirlInward(self,child,field,thetamax =70/360.0*2*np.pi,nbin=50,verbose=True)
		thetabin, n_isdata[k,:], n_swirlhist[k,:], n_inhist[k,:], n_swirlerr[k,:], n_inerr[k,:] = Cornea.getSwirlInward(flowChild,"orientation",maxangle,nbin)
		# def computeFlux(self,child,births,deaths,thetamax =70/360.0*2*np.pi,verbose=False):
		thetabin_flux, isdata[k,:], velflux[k,:], velav[k,:], velproj[k,:], alpha[k,:], vel2av[k,:], birthcount[k,:], deathcount[k,:]= Cornea.computeFlux(flowChild,birthpos, deathpos,thetamax =args.coneangle/360.0*2*np.pi,verbose=False)
		# and then undo the tilt
		# def unTilt(self,child,axis,rot_angle,births='none',deaths='none'):
		birthpos, deathpos = Cornea.unTilt(flowChild,axis,rot_angle,birthpos,deathpos)
		anyflux = True
		nav+=1
	else:
		print("Couldn't find a central n defect, doing nothing!")
	
	#def redressTiltDefect(self,child,defectpos,debug=True):
	if not defectpos2 == "problem":
		goodconf_v.append(k)
		axis, rot_angle = Cornea.redressTiltDefect(flowChild,defectpos2)
		#write.writeConfigurationVTK(flowChild,'retilted_' +str(k) +'.vtp')
		#thetabin, isdata, swirlhist, inhist, swirlerr, inerr = getSwirlInward(self,child,field,thetamax =70/360.0*2*np.pi,nbin=50,verbose=True)
		thetabin2, v_isdata[k,:], v_swirlhist[k,:], v_inhist[k,:], v_swirlerr[k,:], v_inerr[k,:] = Cornea.getSwirlInward(flowChild,"velocity",maxangle,nbin)
		if not anyflux:
			thetabin_flux, isdata[k,:],velflux[k,:], velav[k,:], velproj[k,:], alpha[k,:], vel2av[k,:], birthcount[k,:], deathcount[k,:]= Cornea.computeFlux(flowChild,birthpos, deathpos,thetamax =args.coneangle/360.0*2*np.pi,verbose=False)
			anyflux = True
		nav+=1
	else:
		print("Couldn't find a central v defect, doing nothing!")
		
	# Return to default: For messy systems, use the central z to measure things
	if not anyflux:
		# no need to actually rotate it, tat is it's default position ...
		#zpos = np.array([0,0,Cornea.geom.R])
		#axis, rot_angle, birthpos, deathpos = Cornea.redressTiltDefect(flowChild,zpos,birthpos,deathpos)
		thetabin, n_isdata[k,:], n_swirlhist[k,:], n_inhist[k,:], n_swirlerr[k,:], n_inerr[k,:] = Cornea.getSwirlInward(flowChild,"orientation",maxangle,nbin)
		thetabin2, v_isdata[k,:], v_swirlhist[k,:], v_inhist[k,:], v_swirlerr[k,:], v_inerr[k,:] = Cornea.getSwirlInward(flowChild,"velocity",maxangle,nbin)
		# def computeFlux(self,child,births,deaths,thetamax =70/360.0*2*np.pi,verbose=False):
		thetabin_flux, isdata[k,:],velflux[k,:], velav[k,:], velproj[k,:], alpha[k,:], vel2av[k,:], birthcount[k,:], deathcount[k,:]= Cornea.computeFlux(flowChild,birthpos, deathpos,thetamax =args.coneangle/360.0*2*np.pi,verbose=False)

data['goodconf_n']=goodconf_n
data['goodconf_v']=goodconf_v
data['defuse_n']=defuse_n
data['defuse_v']=defuse_v
data.update({'vdef':vdefout,'nvdef':nvdefout,'ndef':ndefout,'nndef':nndefout,'n_defpos':n_defpos,'v_defpos':v_defpos})
data.update({'thetabin':thetabin,'n_isdata':n_isdata,'n_swirlhist':n_swirlhist,'n_inhist':n_inhist,'n_swirlerr':n_swirlerr,'n_inerr':n_inerr})
data.update({'v_isdata':v_isdata,'v_swirlhist':v_swirlhist,'v_inhist':v_inhist,'v_swirlerr':v_swirlerr,'v_inerr':v_inerr})
data.update({'thetabin_flux':thetabin_flux,'isdata':isdata,'velflux':velflux,'velav':velav,'velproj':velproj,'alpha':alpha,'vel2av':vel2av,'birthcount':birthcount,'deathcount':deathcount})
	
# Save the output files
outcornea='corneastats_testing.p'
pickle.dump(data,open(outcornea,'wb'))

plt.figure()
dtheta=thetabin[1]-thetabin[0]
plt.plot(thetabin+dtheta/2,np.mean(v_swirlhist,axis=0),marker='o',color='r',label='swirl')
plt.plot(thetabin+dtheta/2,np.mean(v_inhist,axis=0),marker='o',color='g',label='inward')
plt.xlabel('theta')
plt.ylabel('swirl,inward')
plt.legend()
plt.title('Swirl and inward for velocity')

plt.figure()
dtheta=thetabin[1]-thetabin[0]
plt.plot(thetabin+dtheta/2,np.mean(n_swirlhist,axis=0),marker='o',color='r',label='swirl')
plt.plot(thetabin+dtheta/2,np.mean(n_inhist,axis=0),marker='o',color='g',label='inward')
plt.xlabel('theta')
plt.ylabel('swirl,inward')
plt.legend()
plt.title('Swirl and inward for orientation')

plt.figure()
dtheta = thetabin_flux[1]-thetabin_flux[0]
plt.plot(thetabin_flux+dtheta/2,np.mean(velflux,axis=0),marker='x',color='m',label='flux')
plt.plot(thetabin_flux+dtheta/2,np.mean(velav,axis=0),marker='o',color='r',label='vav')
plt.plot(thetabin_flux+dtheta/2,np.mean(velproj,axis=0),marker='.',color='k',label='velproj')
plt.plot(thetabin_flux+dtheta/2,np.mean(alpha,axis=0),marker='o',color='g',label='alpha')
plt.plot(thetabin_flux+dtheta/2,np.mean(vel2av,axis=0),marker='o',color='b',label='v2av')
plt.plot(thetabin_flux+dtheta/2,np.mean(birthcount,axis=0),marker='s',color='k',label='birth')
plt.plot(thetabin_flux+dtheta/2,np.mean(deathcount,axis=0),marker='s',color='y',label='death')
plt.xlabel('theta')
plt.ylabel('quantity')
plt.legend()
plt.title('Flux quantities')

plt.show()
	

	
	
