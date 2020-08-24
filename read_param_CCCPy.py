# SAMoSA: Read parameters from a keyword, value type csv file. Made for CCCPy.

import csv


class Param:
    def __init__(self,filename):		
        self.filename=filename

        with open(self.filename, newline='') as csvfile:
            # First read in the keyword: value styled parameter file
            paramreader = csv.reader(csvfile, delimiter=',')
            paramdict = {rows[0]:rows[1] for rows in paramreader}
            print(paramdict)

            # the construct bespoke parameters out of it
            self.N = int(paramdict['N'])
            self.dt = float(parmdict['stepSize'])

            # Geometry
            if paramdict["periodic"] == '0':
                self.periodic = False
                self.constraint = 'plane'
            else:
                self.periodic = True
                self.constraint = 'plane_periodic'
            Lx = float(paramdict['L_x']) 
            Ly = float(paramdict['L_y']) 
            # put into a box to be compatible with samos convention
            self.box = [Lx,Ly,0]
            # and because that appears to be used as well ... cleanup
            self.lx = Lx
            self.ly = Ly


            # elastic potential 
            self.pot_params = {}
            if paramdict["enableHarmonicInterForce"]=='1':
                self.potential = 'soft'
                self.pot_params['k'] = float(paramdict['kharmonic'])
            elif paramdict["enableHertzianInterForce"]=='1':
                self.potential = 'hertzian'
                self.pot_params['k'] = float(paramdict['khertzian'])
            else:
                print("No or unknown particle interactions")
                self.potential = 'none'
                self.pot_params['k'] = 0.0

            # Dynamical parameters (as in 'oneInt' of Samos)
            if paramdict["enableActiveForce"]='1':
                self.v0 = paramdict["v_0"]
            else:
                self.v0 = 0.0
            if 
                self.nu




