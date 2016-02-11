import numpy as np
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import optparse, subprocess
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
from sys import argv
from LE_Utils import save_data
from LE_Utils import FBW_soft as force_x


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
		-X --wallpos	10.0	Position of wall
		-D --Delta		0.0		Width of wall onset in units of X
		-r --nruns		100		Number of runs for each (x0,y0)
		-t --timefac	1.0		Multiply t_max by factor
		-v --verbose	False	Print useful information to screen
		-h --help		False	Print docstring and exit
	
	EXAMPLE
		python LE_LightBoundarySim.py -a 0.5 -r 100 -v
	
	NOTES
	
	BUGS
		-- xmax is poorly constructed
	
	HISTORY
		13 November 2015	Adapted from LE_BoundarySimPlt.py
		01 February 2016	Adopted white noise x variable
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
                  dest="X",default=10.0,type="float")
	parser.add_option('-D','--Delta',
                  dest="Delta",default=0.0,type="float")		
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
	Delta	= opt.Delta
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
	tmax = 1e4*timefac
	
	## Space
	xmax = lookup_xmax(X,a)
	xmin = 0.8*X	## Simulation cutoff
	xinit= 0.9*X	## Particle initial x ("line" IC)
	ymax = 0.5
	
	## Histogramming; xbins and ybins are bin edges.
	Nxbin = 100
	Nybin = 50
	xbins = calculate_xbin(xinit,X,xmax,Nxbin)#np.linspace(xmin,xmax,Nxbin+1)#
	ybins = np.linspace(-ymax,ymax,Nybin+1)
		
	## Initial conditions
	X0Y0 = np.array([[xinit,y0] for y0 in ybins])
	Nparticles = Nybin*Nrun
	if vb: print me+"initial condition injection line; computing",Nparticles,"trajectories"
	
	## Filename; directory and file existence; readme
	hisfile = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+"X"+str(X)+"D"+str(Delta)+"r"+str(Nrun)+\
			"/BHIS_a"+str(a)+"_X"+str(X)+"_D"+str(Delta)+"_r"+str(Nrun)+"_dt"+str(dt)
	check_path(hisfile, vb)
	create_readme(hisfile, vb)
		
	## ----------------------------------------------------------------
	
	## Precompute exp(-t) and initialise histogram
	expmt = 1 if a==0 else np.exp(-np.arange(0,tmax,dt)/(a*a))
	H = np.zeros((Nxbin,Nybin))
	## Loop over initial y-position
	for x0y0 in X0Y0:
		for run in xrange(Nrun):
			## x, y are coordinates as a function of time
			x, y = boundary_sim(x0y0, a, X, Delta, xmin, tmax, expmt, vb)
			h = np.histogram2d(x,y,bins=[xbins,ybins],normed=False)[0]
			H += h*histogram_weight(x0y0[1],y[-1], a)
	H = (H.T)[::-1]
	## When normed=False, need to divide by the bin area
	H /= np.outer(np.diff(ybins),np.diff(xbins))
	## Normalise by number of particles
	H /= Nparticles
	save_data(hisfile, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return hisfile
	
## ====================================================================

def boundary_sim(x0y0, a, X, D, xmin, tmax, expmt, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	Dynamically adds more space to arrays.
	Same results if end of array not hit.
	Harder to test other case: random numbers. However, it looks close enough.
	"""
	me = "LE_BoundarySimPlt.boundary_sim: "
	t0 = time.time()
	
	## Initialisation
	x0,y0 = x0y0
	nstp = int(tmax/dt)
	exstp = nstp/10
	
	## Simulate eta
	if vb: t0 = time.time()
	y = sim_eta(y0, expmt, nstp, a)
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	## Variable of interest
	if vb: t0 = time.time()
	x = np.zeros(nstp); x[0],xt = x0,x0; i,j = 1,0
	## Euler steps to calculate x(t)
	while xt > xmin:
		xt = x[i-1] + dt*(force_x(x[i-1],1.0,X,D) + 1.0*y[i-1])
		x[i] = xt; i +=1
		## Extend array if necessary
		if i == len(x):
			x = np.append(x,np.zeros(exstp))
			y = np.append(y,sim_eta(y[-2],expmt[:exstp],exstp))
			j += 1
	if j>0: print me+"trajectory array extended",j,"times."
	if vb: print me+"Simulation of x",round(time.time()-t0,1),"seconds for",len(x),"steps"
	
	## Clip trailing zeroes from y and x
	x, y = x[:i+1], y[:i+1]	
	return np.vstack([x,y])

## ----------------------------------------------------------------------------	
	
def sim_eta(et0, expmt, npoints, a=1.0):
	"""
	Any alpha-dependence in expmt already taken care of.
	See notes 02/02/2016 for LE / FPE statement.
	"""
	xi = np.sqrt(2) * np.random.normal(0, 1, npoints)
	if a==0:
		return xi
	else:
		return et0*expmt + (1/(a*a))*dt*fftconvolve(expmt,np.append(np.zeros(npoints),xi),"full")[npoints-1:-npoints]
	
## ====================================================================

def histogram_weight(yi,yf,a):
	"""
	Weights depend on starting position and finishing position.
	Probability of y obeys Gaussian with zero mean and variance ???
	"""
	return np.exp(-0.5*a*a*(yi*yi+yf*yf))
	
## ====================================================================

def lookup_xmax(X,a):
	"""
	2015/12/08 NEW DEFINITIONS
	Lookup table for xmax
	Limited testing; X=10.0 only, up to a=1.0
	"""
	if   a<=0.05:	xmax=1.4*X
	elif a<=0.1:	xmax=1.15*X
	elif a<=0.2:	xmax=1.04*X
	elif a<=0.3:	xmax=1.02*X
	elif a<=0.4:	xmax=1.005*X
	elif a<=0.5:	xmax=1.003*X
	elif a<=0.6:	xmax=1.003*X
	elif a<=0.7:	xmax=1.003*X
	elif a<=0.8:	xmax=1.002*X
	elif a<=0.9:	xmax=1.002*X
	else:			xmax=1.001*X
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

def check_path(hisfile, vb):
	"""
	Check whether directory exists; and if existing file will be overwritten.
	"""
	me = "LE_LignBoundarySim.check_path: "
	if os.path.isfile(hisfile):
		raise IOError(me+"file",hisfile,"already exists. Not overwriting.")
	try:
		assert os.path.isdir(os.path.dirname(hisfile))
	except AssertionError:
		os.mkdir(os.path.dirname(hisfile))
		if vb: print me+"Created directory",os.path.dirname(hisfile)
	return
	
def create_readme(hisfile, vb):
	"""
	If no readme exists, make one.
	NOTE commit is the LAST COMMIT -- maybe there have been changes since then.
	Assumes directory exists.
	"""
	me = "LE_LignBoundarySim.create_readme: "
	readmefile = os.path.dirname(hisfile)+"/README.txt"
	try:
		assert os.path.isfile(readmefile)
	except AssertionError:
		now = str(datetime.now().strftime("%Y-%m-%d %H.%M"))
		execute = " ".join(argv)
		commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
		header = "Time:\t"+now+"\nCommit hash:\t"+commit+"\n\n"
		with open(readmefile,"w") as f:
			f.write(header)
		if vb: print me+"Created readme file "+readmefile
	return

## ====================================================================
## ====================================================================
if __name__=="__main__": main()