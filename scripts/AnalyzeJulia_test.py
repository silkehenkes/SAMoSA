# Basic debugging analysis for CCCPy

import sys
import argparse
import pickle


sys.path.insert(1,'/Users/silke/Documents/Coding/SAMoSA/')


from Dynamics import *
from read_param_Julia import *

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", type=str, default="/Users/silke/Documents/Coding/JAMs/for_SAMoSA/simdata/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=0, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=200, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=1, help="step snapshots with this spacing for long calculations")
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
JuliaTest = Dynamics(initype="fromJAMsHDF5",param=param,datapath=args.directory,multiopt="many")

# Now read in as desired
# def readDataManyJAMsHDF5(self,filename0="raw_data.h5")
JuliaTest.readDataManyJAMsHDF5()
JuliaTest.validate_initialise()
# check
JuliaTest.printDiagnostic()

# Start collecting all of this in data dictionary
data={'configuration':args.directory,'skip':args.skip,'howmany':args.howmany}

# 1st basic set of analyses
# Will save as pickle file regardless

# start with basic Vicsek model info
print("Vicsek order parameters")
dirvicsek = np.zeros((JuliaTest.Nsnap,3))
velvicsek = np.zeros((JuliaTest.Nsnap,3))
# angle between n and v from cross product, averaged spatially. Basic chiral measure
singamma = np.zeros((JuliaTest.Nsnap))
abssingamma = np.zeros((JuliaTest.Nsnap))
#tval=np.linspace(0,JuliaTest.Nsnap*JuliaTest.param.dt*JuliaTest.param.dump['freq'],num=JuliaTest.Nsnap)
for l in range(0,JuliaTest.Nsnap):
    dirvicsek[l,:], velvicsek[l,:] = JuliaTest.getVicsek(l)
    singamma[l],abssingamma[l]= JuliaTest.getChiral([1],l)
data["dirvicsek"]=dirvicsek
data["velvicsek"]=velvicsek
data["singamma"]=singamma
data["abssingamma"]=abssingamma


# B - Mean square displacement
#tval, msd = Cells.getMSD(self,takeDrift,usetype='all',verbose=True):
print("Mean square displacement")
tval, msdND = JuliaTest.getMSD(True,usetype=[1],verbose=plotall)
data['tval']=tval
data['msdND']=msdND
tval, msd = JuliaTest.getMSD(False,usetype=[1],verbose=plotall)
data['msd']=msd
# C - Velocity temporal autocorrelation function
# xval, self.velauto, v2av = getVelAuto(self,usetype='all',verbose=True)
print("Velocity autocorrelation function")
tval2, velauto, v2av = JuliaTest.getVelAuto(usetype=[1],verbose=plotall)
data['tval2']=tval2
data['velauto']=velauto
data['v2av']=v2av
# Director autocorrelation funtion
tval2, dirauto,omega,absomega = JuliaTest.getDirAuto(usetype=[1],verbose=plotall)
data['dirauto']=dirauto
data['v2av']=v2av
data['omega']=omega
data['absomega']=absomega

tval2, dispauto, disp2av=JuliaTest.getDispAuto(usetype=[1],verbose=plotall)
data['dispauto']=dispauto
data['disp2av']=v2av

# D - Self-intermediate scattering function,
# 1/N <\sum_n exp(iq[r_n(t)-r_n(0)]>_t,n
#tval, SelfInt2 = SelfIntermediate(self,qval,takeDrift,usetype='all',verbose=True):
# qval is a 3d vector, typical number: 2pi/radius
# Taking this along x without loss of generality. Rethink for actual wounded systems.
print("Self-intermediate function")
# be a bit smarter here. do one with drift removed, once without drift removed, and average radially.

nang=6
angs=np.linspace(0,2*np.pi*(1-1.0/nang),nang) 
print(angs)
SelfInt=np.zeros((JuliaTest.Nsnap,))
SelfIntND=np.zeros((JuliaTest.Nsnap,))
for ang in angs:
    print(ang)
    qval = 2*np.pi/1.0*np.array([np.cos(ang),np.sin(ang),0])
    #def SelfIntermediate(self,qval,takeDrift,usetype='all',verbose=False):
    tval3, SelfIntND0 = JuliaTest.SelfIntermediate(qval,True,usetype=[1],verbose=False)
    SelfIntND+=SelfIntND0
    tval3, SelfInt0 = JuliaTest.SelfIntermediate(qval,False,usetype=[1],verbose=False)
    SelfInt+=SelfInt0
SelfIntND/=len(angs)
SelfInt/=len(angs)
data['qval']=qval
data['tval3']=tval3
data['SelfInt']=SelfInt
data['SelfIntND']=SelfIntND
if plotall:
    plt.figure()
    plt.semilogx(tval,SelfInt,'-r',label='with drift')
    plt.semilogx(tval,SelfIntND,'-k',label='no drift')
    plt.xlabel('time')
    plt.ylabel('Self-intermediate')
    plt.legend()




plt.show()
