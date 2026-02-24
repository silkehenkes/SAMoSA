# SAMoSA: Read parameters from a hdf5 julia file

import h5py as h5
import numpy as np


class Param:
	def __init__(self,filename):		
		self.filename=filename

		rdata = h5.File(filename, 'r')
		self.constraint = 'plane_periodic'
		self.box = list(rdata["system"]["sizes"][()])
		self.lx=self.box[0]
		self.ly=self.box[1]
		self.dt = rdata["integration_info"]["dt"][()]
		self.ntypes=len(np.unique(rdata["frames"]["1"]["type"]))

		#raw_data["system"].keys()
		system =rdata["system"]
		self.force_params = {}

		for forcetype in system["forces"]:
			if forcetype not in self.force_params:
				self.force_params[forcetype] = {}

			for force in  system["forces"][forcetype]:
				if force not in self.force_params[forcetype]:
					self.force_params[forcetype][force] = {}
				for field in system["forces"][forcetype][force]:
					self.force_params[forcetype][force][field] = np.array(system["forces"][forcetype][force][field])

		# debugging:
		self.potential='soft'
		self.pot_params={}
		self.pot_params['k']=1.0
		self.dump = {}
		self.dump['freq'] = rdata['integration_info']["Tsave"][()]
                
		rdata.close()
			
			
			
				
			
				
				
			
			
				
				




