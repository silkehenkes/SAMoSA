# Basic debugging analysis for CCCPy

import sys
import argparse
import pickle

sys.path.insert(1,'/home/sh18581/Documents/SAMoSA/')

from Dynamics import *
from Hessian import * 
from Writer import *
from read_param_CCCPy import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conffile", type=str, default="ParameterList_Readable.csv", help="configuration file")
parser.add_argument("-d", "--directory", type=str, default="/home/sh18581/Documents/Crowds/testing/Demonstration/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=300, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=500, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=1, help="step snapshots with this spacing")

args = parser.parse_args()

# Use the Configuration constructor in its readCSV format to generate the topology 
# fromCSV(kwargs["parampath"],kwargs["datapath"],kwargs["multiopt"])
parampath0 = args.directory + args.conffile
param = Param(parampath0)
Crowd = Dynamics(initype="fromCSV",param=param,datapath=args.directory+'ParticleData/',multiopt="many")

# Now read in as desired
# def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
Crowd.readDataMany("CCCPy",args.skip,args.step,args.howmany,False,readtypes = 'all')

# Generate the average to feed to the Hessian
averageChild = Crowd.makeAverageChild('all')
#print("Checking for wrap: " + str(averageChild.rval[:,0]))
# Feed it to the Hessian
# def __init__(self,conf,rattlers=[],debug=False):
Hess = Hessian(averageChild,rattlers=[],debug=True)
# def makeMatrix(self,dim=2,fixBorder=True,btype = 2,typeweights = 'none',addRestoring=True,ksurf=10.0,addCurvature=False):
# Mass ratio as prefactor for Hessian masses. Using heavy 1000 factor to supress motion of boundary particles
wei = param.mass_radius_ratio
Hess.makeMatrix(2,True,2,[wei,1e6*wei])
Hess.getModes()
Hess.plotModes2d([0,1,2,3,4,5,1000,1100,1200])
proj,projv,proj2av,projv2av,transcomp,velcomp = Crowd.projectModes2d(Hess)
Crowd.plotProjections(Hess,nmodes=50)

plt.show()
