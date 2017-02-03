me0 = "LE_CJSim"

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
from LE_CSim import eul, force_dlin, force_mlin, force_clin, force_ulin

import warnings
# warnings.filterwarnings("error")


##=============================================================================
##=============================================================================

def input():
	"""
	Adapted from LE_CSim 02/02/2017.
	
	PURPOSE
		Simulate coloured noise trajectories in a Cartesian geometry.
		Periodic in the y-direction.
		Save file to be turned into a current quiverplot.
		
	INPUT
	
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
	
	## Undulating wall, one at R and the other at -R.
	if ftype=="ulin":
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
	
	Nxbin = int(50 * (xmax-xmin))		###
	Nybin = int(50 * (ymax-ymin))
	xbins = np.linspace(xmin,xmax,Nxbin+1)
	ybins = np.linspace(ymin,ymax,Nybin+1)
	
	vmax = 4/np.sqrt(a) if a!=0 else 4/np.sqrt(dt)
	Nvbin = 60
	vxbins = np.linspace(-vmax,+vmax,Nvbin+1)
	vybins = np.linspace(-vmax,+vmax,Nvbin+1)
	
	bins = [xbins, ybins, vxbins, vybins]
	
	x = 0.5*(xbins[:-1]+xbins[1:])
	y = 0.5*(ybins[:-1]+ybins[1:])
	vx = 0.5*(vxbins[:-1]+vxbins[1:])
	vy = 0.5*(vybins[:-1]+vybins[1:])
	
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
	hisfile = "CURR_CAR_"+fstr+"_a"+str(a)+"_R"+str(R)+"_S"+str(S)+filepar+"_dt"+str(dt)
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
	
#	## Save bins
#	np.savez(hisdir+binfile,xbins=xbins,ybins=ybins,vxbins=vxbins,vybins=vybins)

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
		coords = simulate_trajectory([xini[i],yini[i]], eIC[i], a, xy_step, dt, tmax, expmt, vb)
		
		## Apply BCs where appropriate
		## Casimir
		if ftype[0]=="c":
			coords[0] = np.abs(coords[0])	## Reflecting BC at x=0
		## Undulating
		if ftype[0]=="u":
			coords = np.array(coords)
			coords[:,coords[0]<0.0] *= -1
			coords[1] %= T
			coords = coords.tolist()
		
		vels = [np.ediff1d(coords[0],to_begin=0.0)/dt, np.ediff1d(coords[1],to_begin=0.0)/dt]
		
		## Histogram. For each bin in x,y
		ti=time.time()
		H += np.histogramdd([coords[0],coords[1],vels[0],vels[1]],bins=bins,normed=False)[0]
		print me+"Histogram %.1f seconds."%(time.time()-ti)
		
	## Divide by bin area and number of particles
	binc = [np.diff(b) for b in bins]
 	H /= reduce(np.multiply, np.ix_(*binc))	
	H /= Nparticles
	
	## Compute average velocity as function of space
	t1=time.time()
	Vx = np.trapz(np.trapz(H, vy, axis=3)*vx, vx, axis=2)
	Vy = np.trapz(np.trapz(H*vy, vy, axis=3), vx, axis=2)
	Hxy = np.trapz(np.trapz(H, vy, axis=3), vx, axis=2)
	print me+"Integral %.1f seconds."%(time.time()-t1)
	
	check_path(filepath, vb)
	np.savez(filepath,xbins=xbins,ybins=ybins,vxbins=vxbins,vybins=vybins,
				Hxy=Hxy, Vx=Vx, Vy=Vy)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return
	
## ====================================================================

def simulate_trajectory(xyini, exyini, a, xy_step, dt, tmax, expmt, vb):
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
	
	return [xy[:,0], xy[:,1]]
		
## ====================================================================
## ====================================================================
if __name__=="__main__":
	input()
