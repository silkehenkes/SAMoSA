
import sys
import argparse
import pickle

sys.path.insert(1,'/data1/pabshettiwar/Simulation_Softwares/SAMOS_ABP/SAMoSA-master/')

from Topology import *
from Writer import *
from read_param import *


directory = "/data1/pabshettiwar/Simulation_Softwares/SAMOS_ABP/tumoroid_system_alignments/1000-morse/pair_nematic/test_auto_run/Seed-6/xi_1.0_J_0.2_dr_0.01_abp-p_0.7_ma_2.50_mD_0.08/"
conffile = "tumoroid-pair_nematic.conf"

step = 1
skip = 0
howmany = 999

# Read the data

parampath0 = directory + conffile
param = Param(parampath0)
write = Writer()

Tumoroid = Topology(initype="fromCSV",param=param,datapath=directory,multiopt="many")
# Now read in as desired
# def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
Tumoroid.readDataMany("SAMoS",skip,step,howmany,False,readtypes = [1],filepattern = 'output')

for k in range(howmany):
    frameChild = Tumoroid.makeFrameChild(k,makeCellList=True)
    # def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False,nuke=True,maxedge=25, coneangle=70.0/360*2*np.pi):
    defects_n, numdefect_n,tess = Tumoroid.getDefects(frameChild,'orientation','nematic',4.0,4,0.8,False,True,False)
    if(k==1):
        print(defects_n, numdefect_n)

    #tess.OrderPatches()
    #write.writePatches(tess,'test_patches_'+str(k)+'.vtp')
    write.writeDefects(defects_n, numdefect_n,'test_polarisationdefects' +str(k) + '.vtp')

