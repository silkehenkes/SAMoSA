
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
parser.add_argument("-d", "--directory", type=str, default="/home/sh18581/Documents/Cornea/radius_300/J_0.1/v0_3.0/nu_1.0/rdivide_0.2/redo_with_n/",help="input directory")
#parser.add_argument("-o", "--output", type=str, help="output directory")
#parser.add_argument("-p", "--prefix", type=str, default="cornea",help="prefix for output file")
parser.add_argument("-s", "--skip", type=int, default=1100, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=30, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=4, help="step snapshots with this spacing in flow field")
parser.add_argument("-a", "--average", type=int, default=5, help="average over these many samples in flow field")
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
v_swirlhist = np.zeros((args.howmany-args.average,nbin))
n_swirlhist = np.zeros((args.howmany-args.average,nbin))
v_inhist = np.zeros((args.howmany-args.average,nbin))
n_inhist = np.zeros((args.howmany-args.average,nbin))
nav = 0
for k in range(args.howmany-args.average):
	useparts, FlowField, PolarField = Cornea.getFlowField(k,args.average)
	flowChild = Cornea.makeFlowChild(k,useparts,FlowField,PolarField)
	# def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False):
	defects_v, numdefect_v,tess = Cornea.getDefects(flowChild,'velocity','polar',20,3,0.5,False,True)
	defects_n, numdefect_n,tess = Cornea.getDefects(flowChild,'orientation','polar',30,3,0.5,False,True)
	
	if output:
		tess.OrderPatches()
		write.writeConfigurationVTK(flowChild,'test_' +str(k) +'.vtp')
		write.writePatches(tess,'test_patches_'+str(k)+'.vtp')
		write.writeDefects(defects_v, numdefect_v,'test_flowdefects_' + str(k) + '.vtp')
		write.writeDefects(defects_n, numdefect_n,'test_polarisationdefects' +str(k) + '.vtp')
	
	# Attempt to compute things now
	#def centralDefect(self,child,defects,numdefect,maxangle=0.3*np.pi):
	defectpos = Cornea.centralDefect(flowChild,defects_n,numdefect_n)
	#def redressTiltDefect(self,child,defectpos,debug=True):
	if not defectpos == "problem":
		flowChild, axis, rot_angle = Cornea.redressTiltDefect(flowChild,defectpos)
		write.writeConfigurationVTK(flowChild,'retilted_' +str(k) +'.vtp')
		#thetabin, isdata, swirlhist, inhist, swirlerr, inerr = getSwirlInward(self,child,field,thetamax =70/360.0*2*np.pi,nbin=50,verbose=True)
		thetabin, isdata, v_swirlhist[k,:], v_inhist[k,:], swirlerr, inerr = Cornea.getSwirlInward(flowChild,"velocity",1.5,nbin)
		thetabin, isdata, n_swirlhist[k,:], n_inhist[k,:], swirlerr, inerr = Cornea.getSwirlInward(flowChild,"orientation",1.5,nbin)
		nav+=1
	else:
		print("Couldn't find a central defect, doing nothing!")

	# Writing output (overkill to start with)
	# No point for delaunay. It will make triangles, even if it has to contort itself into a pretzel.
	#tess.makeEdges()

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



# Save the output files
#outcornea=args.output + args.prefix +'.p'
#pickle.dump(data,open(outcornea,'wb'))
plt.show()

	

	
	
