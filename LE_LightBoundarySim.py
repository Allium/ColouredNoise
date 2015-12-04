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
	tmax = 1e3*timefac
	
	## Space
	xmax = lookup_xmax(X,a)
	xmin = 0.8*X	## Simulation cutoff
	xinit= 0.9*X	## Particle initial x ("line" IC)
	ymax = 0.5
	
	## Histogramming; xbins and ybins are bin edges.
	Nxbin = 200
	Nybin = 50
	xbins = calculate_xbin(xinit,X,xmax,Nxbin)#np.linspace(xmin,xmax,Nxbin+1)#
	ybins = np.linspace(-ymax,ymax,Nybin+1)
		
	## Initial conditions and outfile
	if IC is "line":
		X0Y0 = np.array([[xinit,y0] for y0 in ybins])
		Nparticles = Nybin*Nrun
		if vb: print me+"initial condition injection line; computing",Nparticles,"trajectories"
		hisfile = "Pressure/151203X"+str(X)+"r"+str(Nrun)+\
				"/BHIS_a"+str(a)+"_X"+str(X)+"_ym"+str(ymax)+"_r"+str(Nrun)+"_dt"+str(dt)
	elif IC == "uniform":
		X0Y0 = np.array([[x0,y0] for y0 in ybins for x0 in xbins])
		Nparticles = Nybin*Nxbin
		if vb: print me+"initial condition uniform; computing",Nparticles,"trajectories"
		hisfile = "Pressure/uniform_tight_long/BHIS_a"+str(a)+"_Nx"+str(Nxbin)
	
	## Directory and file existence
	if os.path.isfile(hisfile):
		print me+"file",hisfile,"already exists. Not overwriting."
		raise IOError
	try:
		assert os.path.isdir(os.path.dirname(hisfile))
	except AssertionError:
		print me+"directory",os.path.dirname(hisfile),"doesn't exist. Abort."
		raise AssertionError
	
	## If running a pure wall simulation
	# Nxbin = xbins.size - 1
	
	## ----------------------------------------------------------------
	
	## Precompute exp(-t) and initialise histogram
	expmt = np.exp(-np.arange(0,tmax,dt))
	H = np.zeros((Nxbin,Nybin))
	## Loop over initial y-position
	for x0y0 in X0Y0:
		for run in xrange(Nrun):
			np.random.seed(123)
			## x, y are coordinates as a function of time
			x, y = boundary_sim(x0y0, a, X, FBW, xmin, tmax, expmt, False)
			h = np.histogram2d(x,y,bins=[xbins,ybins],normed=False)[0]
			H += h*histogram_weight(x0y0[1],y[-1])
	H = (H.T)[::-1]
	## When normed=False, need to divide by the bin area -- or do we?
	# H /= np.outer(np.diff(ybins),np.diff(xbins))
	## Normalise by number of particles
	H /= Nparticles
	save_data(hisfile, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return hisfile
	
## ====================================================================
def boundary_sim(x0y0, b, X, f, xmin, tmax, expmt, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin
	
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
	xi = np.sqrt(2) * np.random.normal(0, 1, nstp)
	
	## OU noise
	y = y0*expmt + dt * fftconvolve(expmt,np.append(np.zeros(nstp),xi),"full")[nstp-1:-nstp]
	# y = dt * fftconvolve(expmt,np.append(np.zeros(nstp),xi),"full")[nstp-1:-nstp]
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"

	t0 = time.time()
	## Variable of interest
	x = np.zeros(nstp); x[0] = x0
	## Euler steps to calculate x(t)
	for i in xrange(1,nstp):
		x[i] = x[i-1] + dt*(f(x[i-1],b,X) + y[i-1])
		if x[i] < xmin:	break
	if i==nstp-1: print me+"trajectory array ran out of space. BUG! x="+str(x[-1]),[i];exit()
	if vb: print me+"Simulation of x  ",round(time.time()-t0,1),"seconds for",len(x),"steps"
	
	## Clip y and x to be their actual lengths
	x, y = x[:i+1], y[:i+1]
	
	return np.vstack([x,y])
	
## ----------------------------------------------------------------------------	
## ----------------------------------------------------------------------------	

def boundary_sim_dev(x0y0, b, X, f, xmin, tmax, expmt, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	Dynamically adds more space to arrays.
	Same results if end of array not hit.
	Harder to test other case: random numbers. However, it looks close enough.
	"""
	me = "LE_BoundarySimPlt.boundary_sim_dev: "
	t0 = time.time()
	
	## Initialisation
	x0,y0 = x0y0
	nstp = int(tmax/dt)
	exstp = nstp/10
	
	## Simulate eta
	if vb: t0 = time.time()
	y = sim_eta(y0, expmt, dt, nstp)
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	## Variable of interest
	if vb: t0 = time.time()
	x = np.zeros(nstp); x[0],xt = x0,x0; i,j = 1,0
	## Euler steps to calculate x(t)
	while xt > xmin:
		xt = x[i-1] + dt*(f(x[i-1],b,X) + y[i-1])
		x[i] = xt; i +=1
		if i == len(x):
			x = np.append(x,np.zeros(exstp))
			y = np.append(y,sim_eta(y[-1],expmt,dt,exstp))
			j += 1
	if vb: print me+"trajectory array extended",j,"times."
	if vb: print me+"Simulation of x",round(time.time()-t0,1),"seconds for",len(x),"steps"
	
	## Clip y and x to be their actual lengths
	x, y = x[:i+1], y[:i+1]
	
	return np.vstack([x,y])

## ----------------------------------------------------------------------------	
	
def sim_eta(et0, expmt, dt, npoints):
	xi = np.sqrt(2) * np.random.normal(0, 1, npoints)
	et = et0*expmt + dt * fftconvolve(expmt,np.append(np.zeros(npoints),xi),"full")[npoints-1:-npoints]
	return et
	
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
	## 15/11/30 changed from 1.002 to 1.003
	else:			xmax=1.003*X
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