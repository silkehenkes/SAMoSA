# Basic debugging analysis for CCCPy

import sys
import argparse
import pickle


sys.path.insert(1,'/Users/silke/Documents/Coding/SAMoSA/')


#from Dynamics import *
from Topology import *
from Writer import *
from read_param_Julia import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", type=str, default="/Users/silke/Documents/Coding/JAMs/for_SAMoSA/simdata/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=50, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=20, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=2, help="step snapshots with this spacing for long calculations")
parser.add_argument("-o", "--outfile", type=str, default="oscillations.p", help="pickle file name")
parser.add_argument("--plotall", action='store_true', default=False, help="Plot results for each configuration")
args = parser.parse_args()

# Choose to show as plots as well:
plotall = args.plotall

# Use the Configuration constructor in its readCSV format to generate the topology 
# fromCSV(kwargs["parampath"],kwargs["datapath"],kwargs["multiopt"])
parampath0 = args.directory + "raw_data.h5"
param = Param(parampath0)
# there is an error in the conf file (timestep as naked variable 0.1, actual 0.01, first is not used.). correct time step manually here.
param.dt=0.01
JuliaTest = Topology(initype="fromJAMsHDF5",param=param,datapath=args.directory,multiopt="many")

# Now read in as desired
# def readDataManyJAMsHDF5(self,filename0="raw_data.h5",skip=0,step=1,howmany='all')
JuliaTest.readDataManyJAMsHDF5("raw_data.h5",args.skip,args.step,args.howmany)
#JuliaTest.validate_initialise()
# check
JuliaTest.printDiagnostic()
write = Writer()

for k in range(args.howmany):
    frameChild = JuliaTest.makeFrameChild(k,makeCellList=True)
    # def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False,nuke=True,maxedge=25, coneangle=70.0/360*2*np.pi):
    #defects_n, numdefect_n,tess = JuliaTest.getDefects(frameChild,'orientation','nematic',4.0,4,0.8,False,True,False)
    # no merging of defects (rmerge = 0)
    defects_n, numdefect_n,tess = JuliaTest.getDefects(frameChild,'orientation','nematic',0,4,0.8,False,True,False)
    if(k==1):
        print(defects_n, numdefect_n)

    tess.OrderPatches()
    write.writeConfigurationVTK(frameChild,'test_' +str(k) +'.vtp')
    write.writePatches(tess,'test_patches_'+str(k)+'.vtp')
    write.writeDefects(defects_n, numdefect_n,'test_polarisationdefects' +str(k) + '.vtp')

