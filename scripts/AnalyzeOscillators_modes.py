# Basic debugging analysis for CCCPy

import sys
import argparse
import pickle


sys.path.insert(1,'/data1/shenkes/SAMoSA/')


from Dynamics import *
from read_param import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conffile", type=str, default="obstacle_norm_Dr_0.01_J_0.3_v_0.003.conf", help="configuration file")
parser.add_argument("-d", "--directory", type=str, default="/data1/shenkes/AJ/obstacle_norm_0.002/Dr_0.01/data_J_0.3/data_v0_0.003/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=0, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=500, help="read this many samples")
parser.add_argument("-o", "--outfile", type=str, default="modes_analysis.p", help="pickle file name")
parser.add_argument("--plotall", action='store_true', default=False, help="Plot results for each configuration")
args = parser.parse_args()

# Choose to show as plots as well:
plotall = args.plotall

# Use the Configuration constructor in its readCSV format to generate the topology 
# fromCSV(kwargs["parampath"],kwargs["datapath"],kwargs["multiopt"])
parampath0 = args.directory + args.conffile
param = Param(parampath0)
# there is an error in the conf file (timestep as naked variable 0.1, actual 0.01, first is not used.). correct time step manually here.
param.dt=0.01
Oscillators = Dynamics(initype="fromCSV",param=param,datapath=args.directory,multiopt="many")

# Now read in as desired
# def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
Oscillators.readDataMany("SAMoS",args.skip,1,args.howmany,False,readtypes = 'all')
Oscillators.validate_initialise()
# check
Oscillators.printDiagnostic()

# Start collecting all of this in data dictionary
data={'configuration':args.directory,'skip':args.skip,'howmany':args.howmany}

# 1st basic set of analyses
# Will save as pickle file regardless

averageChild = Oscillators.makeAverageChild('all')
hess = Hessian(averageChild,[],args.plotall)
# def makeMatrix(self,dim=2,fixBorder=True,btype = 2,typeweights = 'none',pairstiff='auto',addRestoring=True,ksurf=10.0,addCurvature=False):
glue=1e6
hess.makeMatrix(2,False,1,[1/np.pi,glue/np.pi],[[1.0,5.0],[5.0, 1.0]],False,10,False)
#hess.makeMatrix(2,False,1,'none',[[1.0,5.0],[5.0, 1.0]],False,10,False)
Qpart,wx,wy=hess.getModes()
data['numtypes']=hess.numtypes
data['glue']=glue
data['Qpart']=Qpart
data['eigval']=hess.eigval

if args.plotall:
    plt.figure()
    plt.plot(hess.eigval,Qpart)
    hess.plotModes2d(range(9))

proj,projv,proj2av,projv2av=Oscillators.projectModes2d(hess)
if args.plotall:
    energy=Oscillators.plotProjections(hess,200)
# keep only low mode projections
data['proj']=proj
data['projv']=projv
data['proj2av']=proj2av
data['projv2av']=projv2av
	  
pickle.dump(data,open(args.directory + args.outfile,'wb'))

plt.show()
