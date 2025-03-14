# ***************************************************************************
# *
# *  Copyright (C) 2013-2016 University of Dundee
# *  All rights reserved. 
# *
# *  This file is part of SAMoS (Soft Active Matter on Surfaces) program.
# *
# *  SAMoS is free software; you can redistribute it and/or modify
# *  it under the terms of the GNU General Public License as published by
# *  the Free Software Foundation; either version 2 of the License, or
# *  (at your option) any later version.
# *
# *  SAMoS is distributed in the hope that it will be useful,
# *  but WITHOUT ANY WARRANTY; without even the implied warranty of
# *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# *  GNU General Public License for more details.
# *
# *  You should have received a copy of the GNU General Public License
# *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# *
# *****************************************************************************
#
# Cleaned up and moved on to Python3 as part of SAMoSA, August 2020 (C) University of Bristol
#
# Major rewrite using pandas dataframes. Preserving samos compatibility (to an extent ... data type going into configuration is now a pandas dataframe. Same keys, however.)

# Reads in data files. Base handler for data analysis

import pandas
import gzip

class ReadData:
  
	def __init__(self, filename,dialect):
		if filename.split('.')[-1] == 'gz':
			self.datafile = gzip.open(filename, newline='')
		else:
			self.datafile = open(filename, newline='')
		self.dialect = dialect
		self.__read_data()
		
	# Read data using pandas. Simplify data structure for Configuration
	def __read_data(self):
		# This outputs a dataframe as is
		if self.dialect == "SAMoS":
			# this has a # in front of the header, and I cannot seem to tell python to disregard it (and not come up as 'unnamed')
			# as a result, all the columns labels are shifted one to the left
			self.data = pandas.read_csv(self.datafile,header=0,sep="\s+")
			temp = self.data.columns
			colshift = {}
			for u in range(len(temp)-1): 
				colshift[temp[u]] = temp[u+1]
			self.data.rename(columns = {temp[len(temp)-1]: 'garbage'},inplace=True)
			self.data.rename(columns = colshift,inplace=True,errors="raise")
			#print(self.data.columns)
		elif self.dialect == "CCCPy":
			self.data = pandas.read_csv(self.datafile,header=0)
			# look of the header
			# currTime,xPos,yPos,xVel,yVel,polAngle,polVel,xPol,yPol,rad,glued
			# We need to muck about with the headers to distil this to a unified format
			# Classical samos header:
			#  id  type  flag  radius  x  y  z  vx  vy  vz  nx  ny  nz 
			self.data.rename(columns={"xPos": "x", "yPos": "y", "xVel": "vx", "yVel": "vy", "xPol": "nx", "yPol": "ny", "rad":"radius", "glued":"type"}, inplace=True,errors="raise")
			#print(self.data.columns)
		elif self.dialect == "CAPMD":
			self.data = pandas.read_csv(self.datafile,header=0)
		else:
			print("Unknown data format dialect!")
		

