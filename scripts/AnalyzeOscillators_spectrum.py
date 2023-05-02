# Basic debugging analysis for CCCPy

import sys
import argparse
import pickle


sys.path.insert(1,'/data1/shenkes/SAMoSA/')


from Dynamics import *
from read_param import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conffile", type=str, default="obstacle_norm_Dr_0.01_J_0.3_v_0.003.conf", help="configuration file")
parser.add_argument("-d", "--directory", type=str, default="/data1/shenkes/AJ/obstacle_norm_0.0/Dr_0.01/data_J_0.3/data_v0_0.003/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=0, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=500, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=20, help="step snapshots with this spacing for long calculations")
parser.add_argument("-o", "--outfile", type=str, default="oscillations.p", help="pickle file name")
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

# start with basic Vicsek model info
print("Vicsek order parameters")
dirvicsek = np.zeros((Oscillators.Nsnap,3))
velvicsek = np.zeros((Oscillators.Nsnap,3))
# angle between n and v from cross product, averaged spatially. Basic chiral measure
singamma = np.zeros((Oscillators.Nsnap))
abssingamma = np.zeros((Oscillators.Nsnap))
#tval=np.linspace(0,Oscillators.Nsnap*Oscillators.param.dt*Oscillators.param.dump['freq'],num=Oscillators.Nsnap)
for l in range(0,Oscillators.Nsnap):
    dirvicsek[l,:], velvicsek[l,:] = Oscillators.getVicsek(l)
    singamma[l],abssingamma[l]= Oscillators.getChiral([1],l)
data["dirvicsek"]=dirvicsek
data["velvicsek"]=velvicsek
data["singamma"]=singamma
data["abssingamma"]=abssingamma


# B - Mean square displacement
#tval, msd = Cells.getMSD(self,takeDrift,usetype='all',verbose=True):
print("Mean square displacement")
tval, msdND = Oscillators.getMSD(True,usetype=[1],verbose=plotall)
data['tval']=tval
data['msdND']=msdND
tval, msd = Oscillators.getMSD(False,usetype=[1],verbose=plotall)
data['msd']=msd
# C - Velocity temporal autocorrelation function
# xval, self.velauto, v2av = getVelAuto(self,usetype='all',verbose=True)
print("Velocity autocorrelation function")
tval2, velauto, v2av = Oscillators.getVelAuto(usetype=[1],verbose=plotall)
data['tval2']=tval2
data['velauto']=velauto
data['v2av']=v2av
# Director autocorrelation funtion
tval2, dirauto,omega,absomega = Oscillators.getDirAuto(usetype=[1],verbose=plotall)
data['dirauto']=dirauto
data['v2av']=v2av
data['omega']=omega
data['absomega']=absomega

tval2, dispauto, disp2av=Oscillators.getDispAuto(usetype=[1],verbose=plotall)
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
SelfInt=np.zeros((Oscillators.Nsnap,))
SelfIntND=np.zeros((Oscillators.Nsnap,))
for ang in angs:
    print(ang)
    qval = 2*np.pi/1.0*np.array([np.cos(ang),np.sin(ang),0])
    #def SelfIntermediate(self,qval,takeDrift,usetype='all',verbose=False):
    tval3, SelfIntND0 = Oscillators.SelfIntermediate(qval,True,usetype=[1],verbose=False)
    SelfIntND+=SelfIntND0
    tval3, SelfInt0 = Oscillators.SelfIntermediate(qval,False,usetype=[1],verbose=False)
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
    plt.ylabel('Self-intermediat')
    plt.legend()


# Now moving to single-time spatial correlations. Compute on an evenly spaced subset of the simulation.

# G - velocity correlation function in Fourier space
velcorrFourier = np.zeros((69,))
# H - real space velocity correlation function ('swirlyness')
velcorrReal = np.zeros((100,))

count=0
for u in range(0,args.howmany,args.step):
    print(u)
    # G - Fourier space velocity correlation function
    # #qrad,valrad,Sqrad=FourierTransVel(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=True)
    qmax = 2*np.pi/2.0
    qrad2,valrad0,Sqrad=Oscillators.FourierTransVel(qmax=qmax,whichframe=u,usetype=[1],L="default",verbose=False)
    velcorrFourier[:len(qrad2)] += Sqrad

    # H - Real space velocity correlation function
    # bins,velcorr = getVelcorrSingle(self,dx,xmax,whichframe=1,usetype='all',verbose=True):
    # spacing < 1 cell radius, out to 50 cell radii
    spacebins,velcorr = Oscillators.getVelcorrSingle(0.5,50,whichframe=u,usetype=[1],verbose=False)
    velcorrReal[:len(spacebins)] += velcorr

    count+=1
# Normalise correctly

velcorrFourier/=count
velcorrReal/=count
data['qrad2']=qrad2
data['velcorrFourier']=velcorrFourier[:len(qrad2)]
data['spacebins']=spacebins
data['velcorrReal']=velcorrReal

# our full structure factor ...
#qmax = 2*np.pi/2.0
qmax = 1.0
#omegamax = 2*np.pi/2.0 # dt spacing is 1, but remember signal analysis
omegamax = 2*np.pi/8.0 # adjust to make feasible
#### CHANGE TO REMOVE DEBUG
#omegamax = 2*np.pi/40.0 # adjust for debug
# velocity
qrad, omega,Velspectrum=Oscillators.getDynStruct(qmax,omegamax,'velocity',[1],L="default",verbose=plotall)
data['qrad']=qrad
data['omega']=omega
data['Velspectrum']=Velspectrum
# displacements
qrad, omega,Dispspectrum=Oscillators.getDynStruct(qmax,omegamax,'displacement',[1],L="default",verbose=plotall)
data['Dispspectrum']=Dispspectrum
# directors
qrad, omega,Dirspectrum=Oscillators.getDynStruct(qmax,omegamax,'director',[1],L="default",verbose=plotall)
data['Dirspectrum']=Dirspectrum

# # Now save all the data as a pickle file
pickle.dump(data,open(args.directory + args.outfile,'wb'))

if plotall:
	
	plt.figure()
	plt.semilogy(qrad2,velcorrFourier[:len(qrad2)],'.-r',lw=2)
	plt.xlabel('q')
	plt.ylabel('Correlation')
	plt.title('Fourier space velocity correlation')
	
	plt.figure()
	plt.plot(spacebins,velcorrReal[:len(spacebins)],'.-r',lw=2)
	plt.xlabel('r')
	plt.ylabel('Correlation')
	plt.title('Real space velocity correlation')
	
# Now save all the data as a pickle file



plt.show()
