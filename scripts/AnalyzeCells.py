# Basic debugging analysis for CCCPy

import sys
import argparse
import pickle

sys.path.insert(1,'/home/sh18581/Documents/SAMoSA/')

from Dynamics import *
from read_param import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--conffile", type=str, default="plane_periodic.conf", help="configuration file")
parser.add_argument("-d", "--directory", type=str, default="/home/sh18581/Documents/Dome/samos_check/periodic_test/data_phi_1/data_v0_0.01/data_nu_0.01/",help="input directory")
parser.add_argument("-s", "--skip", type=int, default=300, help="skip this many samples")
parser.add_argument("-m", "--howmany", type=int, default=500, help="read this many samples")
parser.add_argument("-t", "--step", type=int, default=20, help="step snapshots with this spacing")
parser.add_argument("-o", "--outfile", type=str, default="correlations.p", help="pickle file name")
args = parser.parse_args()

# Choose to show as plots as well:
plotall = True

# Use the Configuration constructor in its readCSV format to generate the topology 
# fromCSV(kwargs["parampath"],kwargs["datapath"],kwargs["multiopt"])
parampath0 = args.directory + args.conffile
param = Param(parampath0)
Cells = Dynamics(initype="fromCSV",param=param,datapath=args.directory,multiopt="many")

# Now read in as desired
# def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
Cells.readDataMany("SAMoS",args.skip,1,args.howmany,False,readtypes = 'all')
Cells.validate_initialise()

# Start collecting all of this in data dictionary
data={'configuration':args.directory,'skip':args.skip,'howmany':args.howmany,'step':args.step}

# 1st basic set of analyses
# Will save as pickle file regardless

# A - Velocity distributions and mean velocity
# vav, vdist,vdist2 = getVelDist(self,bins,bins2,usetype='all',verbose=True):
# Bins are in normalised units (by mean velocity)
velbins=np.linspace(0,3,100)
velbins2=np.linspace(-3,3,100)
vav, vdist,vdist2 = Cells.getVelDist(velbins,velbins2,verbose=plotall)
# Start collecting these in data dictionary
data['velbins']=velbins
data['velbins2']=velbins2
data['vav']=vav
data['vdist']=vdist
data['vdist2']=vdist2

# B - Mean square displacement
#tval, msd = Cells.getMSD(self,takeDrift,usetype='all',verbose=True):
tval, msd = Cells.getMSD(True,usetype='all',verbose=plotall)
data['tval']=tval
data['msd']=msd

# C - Velocity temporal autocorrelation function
# xval, self.velauto, v2av = getVelAuto(self,usetype='all',verbose=True)
tval2, velauto, v2av = Cells.getVelAuto(usetype='all',verbose=plotall)
data['tval2']=tval2
data['velauto']=velauto
data['v2av']=v2av

# D - Self-intermediate scattering function,
# 1/N <\sum_n exp(iq[r_n(t)-r_n(0)]>_t,n
#tval, SelfInt2 = SelfIntermediate(self,qval,takeDrift,usetype='all',verbose=True):
# qval is a 3d vector, typical number: 2pi/radius
# Taking this along x without loss of generality. Rethink for actual wounded systems.
qval = 2*np.pi/1.0*np.array([1,0,0])
tval3, SelfInt = Cells.SelfIntermediate(qval,True,usetype='all',verbose=plotall)
data['qval']=qval
data['tval3']=tval3
data['SelfInt']=SelfInt


# Now moving to single-time spatial correlations. Compute on an evenly spaced subset of the simulation.
# E - basic stats
phival=0
pressure=0
zav=0
# F - static structure factor
structurefact = np.zeros((150,))
# G - velocity correlation function in Fourier space
velcorrFourier = np.zeros((150,))
# H - real space velocity correlation function ('swirlyness')
velcorrReal = np.zeros((150,))

count=0
for u in range(0,args.howmany,args.step):
	print(u)
	# E - Basic statistics (mean square velocity, packing fraction, number density, pressur, force moment, mean energy per particle, total enervy, contact number)
	#vel2av, phival,ndensity, pressure,fmoment,energy,energytot,zav= Cells.getStatsBasic(self,frame=1)
	vel2av0, phival0,ndensity0, pressure0,fmoment0,energy0,energytot0,zav0= Cells.getStatsBasic(u)
	phival+=phival0
	pressure+=pressure0
	zav+=zav0
	
	# F - Static structure factor, i.e. the Fourier transform of g(r). 
	# qrad,valrad = FourierTrans(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=True):
	qmax = 2*np.pi/1.0
	qrad,valrad0 = Cells.FourierTrans(qmax=qmax,whichframe=u,usetype='all',L="default",verbose=False)
	structurefact[:len(qrad)] += valrad0
	
	# G - Fourier space velocity correlation function
	#qrad,valrad,Sqrad=FourierTransVel(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=True)
	qrad2,valrad0,Sqrad=Cells.FourierTransVel(qmax=qmax,whichframe=u,usetype='all',L="default",verbose=False)
	velcorrFourier[:len(qrad2)] += Sqrad
	
	# H - Real space velocity correlation function
	# bins,velcorr = getVelcorrSingle(self,dx,xmax,whichframe=1,usetype='all',verbose=True):
	# spacing < 1 cell radius, out to 50 cell radii
	spacebins,velcorr = Cells.getVelcorrSingle(0.5,50,whichframe=u,usetype='all',verbose=False)
	velcorrReal[:len(spacebins)] += velcorr
	
	count+=1
# Normalise correctly
phival/=count
pressure/=count
zav/=count
data['phival']=phival
data['pressure']=pressure
data['zav']=zav

structurefact/=count
velcorrFourier/=count
velcorrReal/=count
data['qrad']=qrad
data['structurefact']=structurefact[:len(qrad)]
data['qrad2']=qrad2
data['velcorrFourier']=velcorrFourier[:len(qrad2)]
data['spacebins']=spacebins
data['velcorrReal']=velcorrReal

# Now save all the data as a pickle file
pickle.dump(data,open(args.outfile,'wb'))

if plotall:
	plt.figure()
	plt.semilogy(qrad,structurefact[:len(qrad)],'.-r',lw=2)
	plt.xlabel('q')
	plt.ylabel('S(q)')
	plt.title('Structure factor')
	
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
	
plt.show()
