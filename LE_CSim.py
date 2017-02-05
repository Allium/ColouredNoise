me0 = "LE_CSim"

import numpy as np
import scipy as sp
from scipy.signal import fftconvolve
import optparse, os, time
from datetime import datetime

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import pyplot as plt

from LE_Utils import check_path, create_readme, save_data
from LE_LightBoundarySim import sim_eta

import warnings
# warnings.filterwarnings("error")


##=============================================================================
##=============================================================================

def input():
	"""
	PURPOSE
		Simulate coloured noise trajectories in a Cartesian geometry.
		Periodic in the y-direction.
		
	INPUT
		-a	--alpha		0.1		Slope of the potential
		-R	--outrad	2.0		Position of outer wall
		-S	--inrad		-1.0	Position of inner wall
		-t 	--timefac	1.0		Multiply t_max by factor
	
	FLAGS
			--ephi		False	Histogram in eta-phi also, for bulk-constant plots
		-v	--verbose	False	Print useful information to screen
		-h	--help		False	Print docstring and exit
		
	TODO
	"""	
	me = me0+".input: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## INPUT OPTIONS
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
		dest="a",default=0.2,type="float")
	parser.add_option('-f','--ftype',
		dest="ftype",default="dlin",type="str")
	parser.add_option('-R','--outrad',
		dest="R",default=-1.0,type="float")
	parser.add_option('-S','--inrad',
		dest="S",default=-1.0,type="float")
	parser.add_option('-T',
		dest="T",default=-1.0,type="float")
	parser.add_option('--dt',
		dest="dt",default=0.01,type="float")
	parser.add_option('-t','--timefac',
		dest="timefac",default=1.0,type="float")
	parser.add_option('-v','--verbose',
		dest="vb",default=False,action="store_true")
	parser.add_option('-h','--help',
		dest="help", default=False, action="store_true")				  
	opts, argv = parser.parse_args()
	if opts.help: print input.__doc__; return
	a		= opts.a
	ftype	= opts.ftype
	R		= opts.R
	S		= opts.S
	T		= opts.T
	dt		= opts.dt
	timefac = opts.timefac
	vb		= opts.vb
	
	if ftype[0] != "u":	assert R>=S>=T, me+"Input must satisfy R>=S>=T."
	if ftype[0] == "c": assert T>=0.0, me+"For Casimir geometry, demand T>=0."
		
	if vb: print "\n==\n"+me+"Input parameters:\n\t",opts
	
	main(a,ftype,R,S,T,dt,timefac,vb)
	
	return

##=============================================================================

def main(a,ftype,R,S,T,dt,timefac,vb):
	"""
	"""
	me = me0+".main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## CHOOSE FORCE, FILENAME, SPACE
	
	xydata = False	## Only interested in x data -- symmetric in y
	ymin, ymax = 0.0, 1.0	## Only relevant when xydata = True
	
	## Double quadratic potential
	if ftype=="dlin":
		## Force
		fxy = lambda xy: force_dlin(xy,R,S)
		## Filename
		fstr = "DL"
		filepar = ""
		## Simulation limits
		xmax = R+4.0
		xmin = S-4.0
	
	## Casimir quadratic potential. Symmetric about x=0.
	elif ftype=="clin":
		## Force
		fxy = lambda xy: force_clin(xy,R,S,T)
		## Filename
		fstr = "CL"
		filepar = "_T%.1f"%(T)
		## Simulation limits
		xmax = R+4.0
		xmin = 0.0
		
	## Single central wall.
	elif ftype=="mlin":
		## Force
		fxy = lambda xy: force_mlin(xy,R,S,T)
		## Filename
		fstr = "ML"
		filepar = "_T%.1f"%(T)
		## Simulation limits
		xmax = +R+4.0
		xmin = -R-4.0
		
	## Single central wall, centred at S with zero bulk.
	elif ftype=="nlin":
		## Force
		fxy = lambda xy: force_nlin(xy,R,S)
		## Filename
		fstr = "NL"
		filepar = ""
		## Simulation limits
		xmax = +R+4.0
		xmin = -R-4.0
		
	## Undulating wall, one at R and the other at -R.
	elif ftype=="ulin":
		## Force
		## R is position of right wall, S is amplitude, T is wavelength
		fxy = lambda xy: force_ulin(xy,R,S,T)
		## Filename
		fstr = "UL"
		filepar = "_T%.1f"%(T)
		## Simulation limits
		xmax = R+4.0
		xmin = 0.0
		ymax = T
		ymin = 0.0
		xydata = True
		
	else:
		raise IOError, me+"check ftype."
	
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS
	
	## Simulation time
	tmax = 5e2*timefac
	
	## Particles	
	Nparticles = 50
	
	## Injection coordinate: random sample from WN distribution
	x = np.linspace(xmin,xmax,1000)
	rhoWN = np.exp(sp.integrate.cumtrapz(fxy([x,0])[0], x, initial=0.0))
	rhoWN /= rhoWN.sum()
	xini = np.random.choice(x, size=Nparticles, p=rhoWN)
	yini = (ymax-ymin)*np.random.random(Nparticles)
	
	## ------------
	## Bin edges: x, etax, etay
	
	Nxbin = int(100 * (xmax-xmin))
	Nybin = int(100 * (ymax-ymin))
	xbins = np.linspace(xmin,xmax,Nxbin+1)
	ybins = np.linspace(ymin,ymax,Nybin+1)
	
	emax = 4/np.sqrt(a) if a!=0 else 4/np.sqrt(dt)
	Nebin = 100
	exbins = np.linspace(-emax,+emax,Nebin+1)
	eybins = np.linspace(-emax,+emax,Nebin+1)
	
	bins = [xbins, ybins] if xydata else [xbins, exbins, eybins]
	
	## ------------
		
	## Initial noise drawn from Gaussian
	if a == 0.0:
		eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0, [Nparticles,2])
	else:
		eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0, [Nparticles,2])
	
	## ----------------------------------------------------------------
	## Integration algorithm
	xy_step = lambda xy, exy: eul(xy, exy, fxy, dt)
	
	## ----------------------------------------------------------------

	## Filename; directory and file existence; readme
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+\
			"_CAR_"+fstr+"_dt"+str(dt)+"/"
	hisfile = "BHIS_CAR_"+fstr+"_a"+str(a)+"_R"+str(R)+"_S"+str(S)+filepar+"_dt"+str(dt)
	binfile = "BHISBIN"+hisfile[4:]
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
	
	## Save bins
	np.savez(hisdir+binfile,xbins=xbins,ybins=ybins,exbins=exbins,eybins=eybins)

	## ----------------------------------------------------------------
	## SIMULATION
	
	if vb: print me+"Computing",Nparticles,"trajectories. Arena: CAR. Force: "+str(ftype)+"."
	
	## Precompute exp(-t)
	if a == 0:
		## Not used in calculations
		expmt = None
	elif a <= 0.1:
		raise ValueError, me+"Must have alpha>0.1"
	else:
		## a is large enough that the exponential is well resolved.
		expmt = np.exp((np.arange(-10*a,dt,dt))/a)
		
	## ----------------------------------------------------------------
	
	## Initialise histogram in space
	H = np.zeros([b.size-1 for b in bins])
	
	## Loop over initial coordinates
	for i in range(Nparticles):
		## Perform run in Cartesian coordinates
		if vb: print me+"Run",i,"of",Nparticles
		coords = simulate_trajectory([xini[i],yini[i]], eIC[i], a, xy_step, dt, tmax, expmt, xydata, vb)
		
		## Apply BCs where appropriate
		if ftype[0]=="c":	## Reflecting BC at x=0
			idx = coords[0]<0.0
			coords[0][idx] *= -1
			coords[1][idx] *= -1
			coords[2][idx] *= -1
#			coords[0] = np.abs(coords[0])
		if ftype[0]=="u":	## Periodic BC
			coords = np.array(coords)
			coords[:,coords[0]<0.0] *= -1
			coords[1] %= T
			coords = coords.tolist()
		
		## Histogram
		H += np.histogramdd(coords,bins=bins,normed=False)[0]
		
	## Divide by bin area and number of particles
	binc = [np.diff(b) for b in bins]
 	H /= reduce(np.multiply, np.ix_(*binc))	
	H /= Nparticles
	
	check_path(filepath, vb)
	save_data(filepath, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return
	
## ====================================================================

def simulate_trajectory(xyini, exyini, a, xy_step, dt, tmax, expmt, xydata, vb):
	"""
	Run the LE simulation from (x0,y0).
	"""
	me = me0+".simulate_trajectory: "
		
	## Initialisation
	x0,y0 = xyini
	nstp = int(tmax/dt)
	
	## Simulate eta
	if vb: t0 = time.time()
	exy = np.vstack([sim_eta(exyini[0], expmt, nstp, a, dt), sim_eta(exyini[1], expmt, nstp, a, dt)]).T
	if vb: print me+"Simulation of eta",round(time.time()-t0,2),"seconds for",nstp,"steps"
				
	## Spatial variables
	if vb: t0 = time.time()
	xy = np.zeros([nstp,2]); xy[0] = [x0,y0]
	
	## Calculate trajectory
	for i in xrange(0,nstp-1):
		xy[i+1] = xy[i] + xy_step(xy[i],exy[i])
	
	if vb: print me+"Simulation of x",round(time.time()-t0,2),"seconds for",nstp,"steps"
	
	return [xy[:,0], xy[:,1]] if xydata else [xy[:,0], exy[:,0], exy[:,1]]
	
## ====================================================================
## INTEGRATION

def eul(xy, exy, fxy, dt):
	"""
	Euler step.
	Basic routine with all dependencies.
	"""
	return dt * ( fxy(xy) + exy )

## ====================================================================
## FORCES
	
def force_dlin(xy,R,S):
	"""
	Double linear force.
	xy.size = 2
	"""
	fx = (S-xy[0])*(xy[0]<S)+(R-xy[0])*(xy[0]>R)
	fy = 0.0
	return np.array([fx,fy])
	
def force_clin(xy,R,S,T):
	"""
	Casimir setup. Symmetric about x=0. Linear force.
	Reflecting is twice as fast as calculating twice.
	A little faster than using np.piecewise.	
	xy.size = 2
	"""
	x = np.abs(xy[0])
	fx = (T-x)*(T<x)*(x<0.5*(T+S))+\
			+(S-x)*(0.5*(T+S)<x)*(x<S)+\
			+(R-x)*(R<x)
	fx *= np.sign(xy[0])
	return np.array([fx,0.0])
	
def force_mlin(xy,R,S,T):
	"""
	Casimir setup with single wall. Linear force.
	Nonzero bulk.
	Confining walls at +/-R. Central wall symmetric between T and S.
	m for middle. Should have been i for interior but here we are.
	The try is in case passed xy=[x,y]. It is slow, but much faster than
		creating the piecewise for a single xy-point.
	"""
	x = xy[0]
	try:
		if x<=-R:				fx = -x-R
		elif T<=x<0.5*(T+S):	fx = -x+T
		elif 0.5*(T+S)<=x<S:	fx = -x+S
		elif R<=x:				fx = -x+R
		else:					fx = 0.0
	except ValueError:	## If array
		fx = np.piecewise(x,
			[x<=-R, (T<=x)&(x<0.5*(T+S)), (0.5*(T+S)<=x)&(x<S), R<=x],
			[lambda y: -R-y, lambda y: T-y, lambda y: S-y, lambda y: R-y])
	return [fx,0.0]
	
def force_nlin(xy,R,S):
	"""
	Casimir setup with single wall. Linear force.
	Zero bulk.
	Confining walls at +/-R. Central wall centread at S.
	"""
	x = xy[0]
	try:
		fx = -x-R if x<=S else -x+R
	except ValueError:	## If array
		fx = (-x-R)*(x<=S) + (-x+R)*(S<x)
	return np.array([fx,0.0])
	
	
def force_ulin(xy,R,A,lam):
	"""
	Undulating. No y-offset yet.
	R is position of right-hand wall.
	A is amplitude of undulation.
	lam is wavelength.
	"""
	x, y = xy
	y = 2*np.pi*(y/lam)
	bi = A*np.sin(y)	## Inner boundary
	try:
		if x<=-R+bi:	fxy = (-x-R+bi)*np.array([1.0,-2*np.pi*A/lam*np.cos(y)])
		elif x>R+bi:	fxy = (-x+R+bi)*np.array([1.0,-2*np.pi*A/lam*np.cos(y)])
		else:			fxy = np.array([0.0,0.0])
	except ValueError:	## If array
		X, Y = np.meshgrid(x, y, indexing="ij")
		bi = A*np.sin(Y)
		vec = np.array([np.ones(X.shape),-2*np.pi/lam*A*np.cos(Y)])
		fxy = np.outer(vec, (+X-R-bi)*(X>+R+bi)) + np.outer(vec, (-X-R+bi)*(X<-R+bi))
	return fxy
	
def potential_ulin(x,y,R,S,T):
	"""
	Potential is quadratic. Different arguments to force_ulin.
	"""
	X, Y = np.meshgrid(x, y, indexing="ij")
	U = 0.5*(X-R-S*np.sin(2*np.pi*Y/T))**2 * (X>+R+S*np.sin(2*np.pi*Y/T)) +\
		0.5*(X+R-S*np.sin(2*np.pi*Y/T))**2 * (X<-R+S*np.sin(2*np.pi*Y/T))
	U -= U.min()
	return U
	
	
## ====================================================================
## ====================================================================
if __name__=="__main__":
	input()
