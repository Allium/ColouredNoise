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
		-I --IC			line	The initial condition
		-v --verbose	False	Print useful information to screen
		-h --help		False	Print docstring and exit
	
	EXAMPLE
		python LE_LightBoundarySim.py -a 0.5 -r 100 -v
	
	NOTES
		Proper treatment of alpha parameter.
		Not sure about weight for uniform IC -- assume it's alright.
		If IC=="uniform", Nrun must be small.
	
	BUGS
		-- xmax is poorly constructed
	
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
	parser.add_option('-X','--wallpos',
                  dest="X",default=1.0,type="float")		
	parser.add_option('-r','--nrun',
                  dest="Nrun",default=100,type="int")
	parser.add_option('--dt',
                  dest="dt",default=0.01,type="float")		
	parser.add_option('-t','--timefac',
                  dest="timefac",default=1.0,type="float")	 
	parser.add_option('-I','--IC',
                  dest="IC",default="line",type="str")	  
	parser.add_option('-v','--verbose',
                  dest="verbose",default=False,action="store_true",
				  help="Print useful information to screen.")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")					  
	opt = parser.parse_args()[0]
	if opt.help: print main.__doc__; return
	a		= opt.potmag
	X		= opt.X
	Nrun	= opt.Nrun
	global dt; dt = opt.dt
	timefac = opt.timefac
	IC		= opt.IC
	vb		= opt.verbose

	assert IC == "line" or "uniform"
	
	if vb: print "\n==  "+me+"a =",a," Nruns =",Nrun," ==\n"

	## ----------------------------------------------------------------
	## Parameters
	
	## Simulation time
	tmax = 5e3*timefac
	
	## Space
	# xmax = calculate_xmax(X,a)
	xmax = lookup_xmax(X,a)
	xmin = 0.7*X	## Simulation cutoff
	xinit= 0.9*X	## Particle initial x ("line" IC)
	ymax = 0.5
	
	## Histogramming
	Nxbin = 200
	Nybin = 50
	xbins = calculate_xbin(xinit,X,xmax,Nxbin)
	ybins = np.linspace(-ymax,ymax,Nybin+1)
		
	## Initial conditions and outfile
	if IC is "line":
		X0Y0 = np.array([[xinit,y0] for y0 in ybins])
		Nparticles = Nybin*Nrun
		if vb: print me+"initial condition injection line; computing",Nparticles,"trajectories"
		hisfile = "Pressure/test/BHIS0.7_a"+str(a)+"_X"+str(X)+"_Nx"+str(Nxbin)+"_r"+str(Nrun)+"_dt"+str(dt)
		# hisfile = "Pressure/151125X"+str(X)+"Nx"+str(Nxbin)+"r"+str(Nrun)+\
				# "/BHIS_a"+str(a)+"_X"+str(X)+"_Nx"+str(Nxbin)+"_r"+str(Nrun)+"_dt"+str(dt)
	elif IC == "uniform":
		X0Y0 = np.array([[x0,y0] for y0 in ybins for x0 in xbins])
		Nparticles = Nybin*Nxbin
		if vb: print me+"initial condition uniform; computing",Nparticles,"trajectories"
		hisfile = "Pressure/uniform_tight_long/BHIS_a"+str(a)+"_Nx"+str(Nxbin)
	
	if os.path.isfile(hisfile):
		print me+"file",hisfile,"already exists. Not overwriting."
		return hisfile
	assert os.path.isdir(os.path.dirname(hisfile))
	
	## ----------------------------------------------------------------
	
	## Precompute exp(-t) and initialise histogram
	expmt = np.exp(-np.arange(0,tmax,dt))
	H = np.zeros((Nxbin,Nybin))
	## Loop over initial y-position
	for x0y0 in X0Y0:
		for run in xrange(Nrun):
			## x, y are coordinates as a function of time
			x, y = boundary_sim(x0y0, a, X, FBW, xmin, tmax, expmt, False)
			h, xbins, ybins = np.histogram2d(x,y,bins=[xbins,ybins], normed=True)
			H += h*histogram_weight(x0y0[1],y[-1])
	H = (H.T)[::-1]
	## Normalise by number of particles
	H /= Nparticles
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
	np.random.seed(1234)
	xi = np.random.normal(0, 1, nstp)
	
	## OU noise
	if expmt is None or expmt.shape[0]!=nstp:
		if vb: print me+"precomputing exp(-t] is much musch faster"
		expmt = np.exp(-np.linspace(0,tmax,nstp))
	# y = y0*expmt + np.sqrt(2)*dt * fftconvolve(expmt,np.append(np.zeros(nstp),xi),"full")[nstp-1:-nstp]
	y = np.sqrt(2)*dt * fftconvolve(expmt,np.append(np.zeros(nstp),xi),"full")[nstp-1:-nstp]
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	t0 = time.time()
	## Variable of interest
	x = np.zeros(nstp); x[0] = x0
	## Euler steps to calculate x(t)
	for i in xrange(1,nstp):
		x[i] = x[i-1] + dt*(f(x[i-1],b,X) + y[i-1])
		if x[i] < xmin:	break	#y *= -1
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
	# xmax = max([(1.0+0.02/(a*a*a)),1.05])*X
	# xmax = min([xmax,3.0*X])
	## 19/11/2015 -- in response to explore_pars
	xmax = X*(1.0+0.02+0.1*(0.1/a))
	## 25/11/2015
	xmax = round(xmax,3)
	return xmax

def lookup_xmax(X,a):
	"""
	Lookup table for xmax
	Limited testing; X=5.0 only
	"""
	if a<=0.1:		xmax=1.2*X
	elif a<=0.2:	xmax=1.1*X
	elif a<=0.3:	xmax=1.04*X
	elif a<=0.4:	xmax=1.04*X
	elif a<=0.5:	xmax=1.005*X
	elif a<=0.6:	xmax=1.005*X
	elif a<=0.7:	xmax=1.004*X
	elif a<=0.8:	xmax=1.004*X
	else:			xmax=1.002*X
	return xmax
	
def calculate_xbin(xinit,X,xmax,Nxbin):
	"""
	Return histogram bins in x
	"""
	# xbins = np.linspace(xinit,xmax,Nxbin+1)
	## Extra bins for detail in wall region
	NxbinL = Nxbin/2; NxbinR = Nxbin - NxbinL
	xbins = np.unique(np.append(np.linspace(xinit,X,NxbinL+1),np.linspace(X,xmax,NxbinR+1)))
	return xbins

## ====================================================================
## ====================================================================
if __name__=="__main__": main()