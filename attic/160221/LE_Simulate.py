import numpy as np
import matplotlib.pyplot as plt
import csv
import os,time
from sys import argv

from LE_Plot import get_filenames


def main():
	"""
	STARTED
	07/03/2015
	
	PURPOSE
	Simulate a Langevin process.
	Output data to file.
	
	INPUTS
	b: force level
	X: wall postion
	
	EXECUTION
	python LE_Simulate.py b X timefac
	
	EXAMPLE
	python LE_Simulate.py 1.0 5.0 1
	
	RETURNS
	[t,x,eta,xi], outfile
	
	BUGS / TODO
	- trapz integration
	"""
	
	## Read parameter values from CLAs
	try: b = float(argv[1])
	except IndexError: b=1.0	
	try: X = float(argv[2])
	except IndexError: X=5.0
	try: timefac = float(argv[3])
	except IndexError: timefac = 1.0
	try: BWpot = int(argv[4])
	except IndexError: BWpot=1
	
	## Run simulation; returns datafile path
	out = LEsim(b,X,BWpot)
	
	return out

##================================================

def LEsim(b=None, X=None, timefac=1.0, BWpot=0):
	"""
	Run the LE simulation.
	"""

	t0 = time.time()
	me = "LE_Simulate.LEsim: "

	## System parameters
	if b is None:
		try: b = float(argv[1])
		except IndexError: b=1.0
	if X is None:
		try: X = float(argv[2])
		except IndexError: X=5.0
	## Choose potential
	if BWpot:	force = FBW; pot = "BW"
	else:		force = FHO; pot = "HO"
	
	## Simulation parameters
	dt = 0.05
	tmax = 1.0*10**4 * (X)*timefac
	nstp = int(tmax/dt)
	tarr = np.arange(0,tmax,dt)

	## Initialisation
	eta0 = 0.0
	x0 = 0.0
	seed = 65438; np.random.seed(seed)
	xi = np.sqrt(2*b)*np.random.normal(0,1,nstp)

	## Output options
	outfile  = get_filenames(b,X,timefac,BWpot)[0]
	savedata = not os.path.isfile(outfile)
	
	## OU noise
	t0=time.time()
	expmt = np.exp(-tarr)	## Precompute exp(-t) array
	eta = eta0*expmt
	eta[0] += xi[0]
	lbound = int(35/dt)	## Set exp(-35) to zero -- only 700 terms
	for i in xrange(1,nstp):
		j = min(i,lbound)
		eta[i] += expmt[:j][::-1].dot(xi[i-j:i])*dt
	print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps";t0 = time.time()

	## Variable of interest
	x = dt*np.cumsum( eta )
	x[0] += x0; fdtsum=0
	for i in xrange(1,nstp):
		fdtsum += dt*force(x[i-1],b,X)
		x[i] += fdtsum
	print me+"Simulation of x  ",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	if savedata:
		save_data(outfile,np.vstack((tarr,x,eta,xi)))

	return np.vstack([tarr,x,eta,xi])
	
	
##================================================
## Potentials
	
def FHO(x,b,X):
	""" Force at position x for harmonic confinement """
	return -b*x

def FBW(x,b,X):
	""" Force at position x for bulk+wall confinement """
	return -b*(np.abs(x)>=X).astype(int)*np.sign(x)
	
	

##==========================================
	
def save_data(outfile,data):
	""" Write .npy file of data. File must read with np.load() """
	
	me = "LE_Simulate.save_data: "
	t0 = time.time()
	np.save(outfile+".npy",data)
	print me+"Data saved",outfile+".npy"
	print me+"Saving data",round(time.time()-t0,1),"seconds"
	
	return

##================================================
##================================================
if __name__=="__main__":
	main()
