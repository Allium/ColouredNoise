import numpy as np
import matplotlib.pyplot as plt
import os, time
import optparse
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
from sys import argv

from LE_Utils import *


def main():
	"""
	NAME
		LE_LightBoundarySim.py
	
	PURPOSE
		Simulate trajectories close to the wall (at +X) in the BW setup
	
	EXECUTION
		python LE_LightBoundarySim.py flags
		
		
	FLAGS
		-a --alpha		0.1		Slope of the potential
		-r --nruns		100		Number of runs for each (x0,y0)
		-t --timefac	1.0		Multiply t_max by factor
		-v --verbose	False	Print useful information to screen
		-h --help		False	Print docstring and exit
	
	EXAMPLE
		python LE_LightBoundarySim.py -a 0.5 -r 100 -v
	
	NOTES
		Proper treatment of alpha parameter
	
	BUGS
		-- xmax is poorly hard-coded
	
	STARTED
		13 November 2015	Adapted from LE_BoundarySimPlt.py
	"""	
	
	me = "LE_LightBoundarySim.main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## Input options
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
                  dest="potmag",default=0.1,type="float",
				  help="The steepness of the potential.")
	parser.add_option('-r','--nrun',
                  dest="Nrun",default=100,type="int")		
	parser.add_option('-t','--timefac',
                  dest="timefac",default=1.0,type="float")	  
	parser.add_option('-v','--verbose',
                  dest="verbose",default=False,action="store_true",
				  help="Print useful information to screen.")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")					  
	opt = parser.parse_args()[0]
	if opt.help: print main.__doc__; return
	a		= opt.potmag
	X		= 1.0
	Nrun	= opt.Nrun
	timefac = opt.timefac
	vb		= opt.verbose

	if vb: print "\n==  "+me+"a =",a," Nruns =",Nrun," ==\n"

	## ----------------------------------------------------------------
	## Parameters
	
	## Simulation	
	x0 = 0.9*X
	tmax = 1e3*timefac	## This is large enough to ensure re-hitting?
	global dt; dt = 0.02
	
	## Histogramming
	xmax = calculate_xmax(X,a)
	xmin = 0.8*X	## For plotting, histogram bins and simulation cutoff
	ymax = 0.5
	Nxbin = 50
	Nybin = 50
	xbins = np.linspace(xmin,xmax,Nxbin+1)
	ybins = np.linspace(-ymax,ymax,Nybin+1)
	dx = (xmax-x0)/Nxbin; dy = 2*ymax/Nybin
	
	hisfile = "Pressure/BHIS/BHIS_a"+str(a)+"_Nx"+str(Nxbin)+"_r"+str(Nrun)
	
	if os.path.isfile(hisfile): return hisfile
	
	## ----------------------------------------------------------------
	
	## Precompute exp(-t) and initialise histogram
	expmt = np.exp(-np.arange(0,tmax,dt))
	H = np.zeros((Nxbin,Nybin))
	## Loop over initial y-position
	for y0 in ybins:
		if vb: print me+str(Nrun),"trajectories with y0 =",round(y0,2)
		for run in xrange(Nrun):
			x, y = boundary_sim((x0,y0), a, X, FBW, xmin, tmax, expmt, False)
			h, xbins, ybins = np.histogram2d(x,y,bins=[xbins,ybins])
			H += h*histogram_weight(y0,y[-1])
	H = (H.T)[::-1]
	## Normalise by number of particles
	H /= Nybin*Nrun
	# H /= np.trapz(np.trapz(H,dx=dx,axis=1), dx=dy)
	save_data(hisfile, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return hisfile
	
## ====================================================================
def boundary_sim(x0y0, b, X, f, xmin, tmax, expmt=None, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<x0
	
	NOTES
	-- Assuming tmax is sufficient time to explore and return.
	If this turns out to not be the case, will have to dynamically increase array size.
	-- Computing eta and then x. Maybe faster to have one loop.
	"""
	me = "LE_BoundarySimPlt.boundary_sim: "
	t0 = time.time()
	
	## Initialisation
	x0,y0 = x0y0
	nstp = int(tmax/dt)
	xi = np.random.normal(0, 1, nstp)
	
	## OU noise
	if expmt is None or expmt.shape[0]!=nstp: expmt = np.exp(-np.linspace(0,tmax,nstp))
	y = y0*expmt + np.sqrt(2)*dt * fftconvolve(expmt,np.append(np.zeros(nstp),xi),"full")[nstp-1:-nstp]
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	t0 = time.time()
	## Variable of interest
	x = np.zeros(nstp); x[0] = x0
	## Euler steps to calculate x(t)
	for i in xrange(1,nstp):
		x[i] += x[i-1] + dt*(f(x[i-1],b,X) + y[i-1])
		if x[i] < xmin:	break
	if i==nstp-1 and vb: print me+"Trarray ran out of space. BUG!"
	if vb: print me+"Simulation of x  ",round(time.time()-t0,1),"seconds for",len(x),"steps"
	
	## Clip y and x to be their actual lengths
	x, y = x[:i+1], y[:i+1]

	return np.vstack([x,y])

## ====================================================================

def histogram_weight(yi,yf):
	"""
	Weights depend on starting position and finishing position.
	Probability of y obeys Gaussian with zero mean and variance ???
	"""
	return np.exp(-(yi*yi+yf*yf)/(2.0))
	
## ====================================================================
def calculate_xmax(X,a):
	xmax = max([X*(1.0+0.02/(a*a*a)),1.05])
	xmax = min([xmax,3.0])
	return xmax

## ====================================================================
## ====================================================================
if __name__=="__main__": main()