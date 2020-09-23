
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
#parser.add_argument("-o", "--output", type=str, help="output directory")
#parser.add_argument("-p", "--prefix", type=str, default="cornea",help="prefix for output file")
parser.add_argument("-s", "--skip", type=int, default=500, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=30, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=4, help="step snapshots with this spacing in flow field")
parser.add_argument("-a", "--average", type=int, default=5, help="average over these many samples in flow field")
parser.add_argument("--coneangle", type=int, default=30, help="cone angle in degrees")
#parser.add_argument("-u", "--maxtype", type=int, default=3, help="Up to what maximum type should I process data?")
#parser.add_argument("--getMSD", action='store_true', default=False, help="Compute mean squared displacement?")
#parser.add_argument("--plot", action='store_true', default=False, help="Plot MSD and correlations")
#parser.add_argument("--ignore", action='store_true', default=False, help="Ignore complications like missing potentials for quick result (warning!)")

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

# Defects
vdefout= []
nvdefout = []
ndefout= []
nndefout = []

# Tilting success
goodconf_n = []
goodconf_v = []

# Central defect positon
n_defpos = []
v_defpos = []

rmerge=10
nuke=True
maxedge=25

data['rmerge']=rmerge
data['tesstype']='delaunay'
data['nuke']=nuke
data['maxedge']=maxedge

# Attempt to compute things now
# Exclude the boundary by stopping a couple of cells short from the cone angle. Cell diameter = 10 (micron)
# 3 cells in radians
dboundary = 30.0/Cornea.geom.R
print(dboundary)
maxangle = args.coneangle/360.0*2*np.pi - dboundary
print('Going out to maximum angle ' + str(maxangle) + ' in degrees ' + str(maxangle/(2*np.pi)*360))

data['maxangle']=maxangle
	
	

nav = 0
for k in range(args.howmany-args.average):
	useparts, FlowField, PolarField = Cornea.getFlowField(k,args.average)
	flowChild = Cornea.makeFlowChild(k,useparts,FlowField,PolarField)
	# def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False):
	defects_v, numdefect_v,tess = Cornea.getDefects(flowChild,'velocity','polar',rmerge,3,0.5,False,True,nuke=True,maxedge=maxedge)
	defects_n, numdefect_n,tess = Cornea.getDefects(flowChild,'orientation','polar',rmerge,3,0.5,False,True,nuke=True,maxedge=maxedge)
	
	vdefout.append(defects_v)
	nvdefout.append(numdefect_v)
	ndefout.append(defects_n)
	nndefout.append(numdefect_n)
	
	if output:
		tess.OrderPatches()
		write.writeConfigurationVTK(flowChild,'test_' +str(k) +'.vtp')
		#write.writePatches(tess,'test_patches_'+str(k)+'.vtp')
		write.writeDefects(defects_v, numdefect_v,'test_flowdefects_' + str(k) + '.vtp')
		write.writeDefects(defects_n, numdefect_n,'test_polarisationdefects' +str(k) + '.vtp')
	
	
	
	# We are redressing the tilt twice. This rewrites the flowChild. Hence we need a copy. Use the actual Configuration (near) copy constructor for once.
	# def makeChild(self,parentconf,frame=1,usetype='all'):
	flowChild2 = Configuration(initype="makeChild",parentconf=flowChild,frame=1,usetype="all")
	
	# Use the director defect to redress tilt for n field
	#def centralDefect(self,child,defects,numdefect,maxangle=0.3*np.pi):
	defectpos = Cornea.centralDefect(flowChild,defects_n,numdefect_n,maxangle=maxangle)
	n_defpos.append(defectpos)
	if not defectpos == "problem":
		goodconf_n.append(k)
		flowChild, axis, rot_angle = Cornea.redressTiltDefect(flowChild,defectpos)
		#write.writeConfigurationVTK(flowChild,'retilted_' +str(k) +'.vtp')
		#thetabin, isdata, swirlhist, inhist, swirlerr, inerr = getSwirlInward(self,child,field,thetamax =70/360.0*2*np.pi,nbin=50,verbose=True)
		thetabin, n_isdata[k,:], n_swirlhist[k,:], n_inhist[k,:], n_swirlerr[k,:], n_inerr[k,:] = Cornea.getSwirlInward(flowChild,"orientation",maxangle,nbin)
		nav+=1
	else:
		print("Couldn't find a central n defect, doing nothing!")
	
	# Same thing but for v
	defectpos2 = Cornea.centralDefect(flowChild2,defects_v,numdefect_v,maxangle=maxangle)
	v_defpos.append(defectpos2)
	#def redressTiltDefect(self,child,defectpos,debug=True):
	if not defectpos2 == "problem":
		goodconf_v.append(k)
		flowChild2, axis, rot_angle = Cornea.redressTiltDefect(flowChild2,defectpos2)
		#thetabin, isdata, swirlhist, inhist, swirlerr, inerr = getSwirlInward(self,child,field,thetamax =70/360.0*2*np.pi,nbin=50,verbose=True)
		thetabin2, v_isdata[k,:], v_swirlhist[k,:], v_inhist[k,:], v_swirlerr[k,:], v_inerr[k,:] = Cornea.getSwirlInward(flowChild,"velocity",maxangle,nbin)
		nav+=1
	else:
		print("Couldn't find a central v defect, doing nothing!")

data['goodconf_n']=goodconf_n
data['goodconf_v']=goodconf_v
data.update({'vdef':vdefout,'nvdef':nvdefout,'ndef':ndefout,'nndef':nndefout,'n_defpos':n_defpos,'v_defpos':v_defpos})
data.update({'thetabin':thetabin,'n_isdata':n_isdata,'n_swirlhist':n_swirlhist,'n_inhist':n_inhist,'n_swirlerr':n_swirlerr,'n_inerr':n_inerr})
data.update({'v_isdata':v_isdata,'v_swirlhist':v_swirlhist,'v_inhist':v_inhist,'v_swirlerr':v_swirlerr,'v_inerr':v_inerr})
	
# Save the output files
outcornea='corneastats.p'
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


plt.show()

	

	
	
