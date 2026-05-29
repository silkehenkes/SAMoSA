import numpy as np
import matplotlib.pyplot as plt

import sys
import argparse
import pickle

#sys.path.insert(1,'/media/chinmay/DATA/Leiden-work/Simulations/SAMoSA-master/')
sys.path.insert(1,'/home/git/SAMoSA/')

from Topology import *
from Writer import *
from read_param import *

def get_defects_configs(directory, conffile= "tumoroid-pair_nematic.conf"):
    #directory = "/disks/pi-henkes/pabshettiwar/1000-morse/pair_polar/phase_diagram/xi_1.0_J_0.8_dr_0.01_abp-p_0.7_ma_2.50_mD_0.08/"

    step = 1
    skip = 100
    howmany = 4
    system_dim = 2
    rmerge = 4

    # Read the data
    parampath0 = directory + conffile
    param = Param(parampath0)
    write = Writer()

    Tumoroid = Topology(initype="fromCSV",param=param,datapath=directory,multiopt="many")
    # Now read in as desired
    # def readDataMany(self,skip=0,step=1,howmany='all',Nvariable=False,readtypes = 'all'):
    Tumoroid.readDataMany("SAMoS",skip,step,howmany,False,readtypes = 'all',filepattern = 'output')

    count_defects_n = []
    count_numdefects = []
    defect_orientations = []

    for k in range(howmany):
        frameChild = Tumoroid.makeFrameChild(k,makeCellList=True)
        # def getDefects(self,child,field,symtype,rmerge = 5, zmin = 4, mult = 0.8,closeHoles=True,delaunay=False,nuke=True,maxedge=25, coneangle=70.0/360*2*np.pi):
        defects_n, numdefect_n,  theta_orients, n_orients, tess = Tumoroid.getDefects(frameChild,'orientation','nematic',rmerge,4, 0.8, True, False,True,False)
        count_defects_n.append(defects_n)
        count_numdefects.append(numdefect_n)
        defect_orientations.append(theta_orients)

    return count_defects_n, count_numdefects, defect_orientations

defects_all, num_defects, defect_orients = get_defects_configs("/media/chinmay/DATA/Leiden-work/Simulations/pair_nematic_activity_data/traceless/kperp_1.0/J_0.1/v0_0.15/", conffile= "tumoroid-pair_nematic.conf")

