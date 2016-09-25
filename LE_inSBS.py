import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import optparse, os, time
from datetime import datetime

from LE_Utils import save_data
from LE_LightBoundarySim import check_path, create_readme, sim_eta

def input():
	"""
	PURPOSE
		Simulate coloured noise trajectories in 2D disc geometry, where the density
		at infinity is fixed and a disc wall is centred at the origin.
		Simplified from LE_SBS.py.
		
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
	me = "LE_inSBS.input: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## INPUT OPTIONS
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
		dest="a",default=0.2,type="float")
	parser.add_option('-S','--inrad',
		dest="S",default=-1.0,type="float")
	parser.add_option('-t','--timefac',
		dest="timefac",default=1.0,type="float")
	parser.add_option("--ephi",
		dest="ephi", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="vb",default=False,action="store_true")
	parser.add_option('-h','--help',
		dest="help", default=False, action="store_true")				  
	opts, argv = parser.parse_args()
	if opts.help: print input.__doc__; return
	a		= opts.a
	S		= opts.S
	timefac = opts.timefac
	ephi	= opts.ephi
	vb		= opts.vb
	
	dt = 0.01
	ftype = "linin"
	intmeth = ""
	
	if vb: print "\n==\n"+me+"Input parameters:\n\t",opts
	
	main(a,ftype,S,dt,timefac,intmeth,ephi,vb)
	
	return

##=============================================================================

def main(a,ftype,S,dt,timefac,intmeth,ephi,vb):
	"""
	"""
	me = "LE_inSBS.main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## CHOOSE FORCE
	if "linin":
		force = lambda xy, r: force_linin(xy,r,S)
		fstr = "L"
		fparstr = ""
	else:
		raise IOError, me+"ftype must be one of {linin}."
	
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS
	
	## Simulation time
	tmax = 5e2*timefac
	
	## Simulation limits
	rmax = S+max(1,4.0*round(np.sqrt(a),0))
	rmin = 0.0
	## Injection coordinate
	rini = S + 1.0
	
	## ------------
	## Bin edges
	
	Nrbin = int(150 * (rmax-rmin))
	rbins = np.linspace(rmin,rmax,Nrbin+1)
	
	Npbin = 50
	pbins = np.linspace(0.0,2*np.pi,Npbin)
	
	ermax = 4/np.sqrt(a) if a!=0 else 4/np.sqrt(dt)
	Nerbin = 150
	erbins = np.linspace(0.0,ermax,Nerbin+1)
	
	if ephi:	
		Nepbin = 50
		epbins = np.linspace(0.0,2*np.pi,Nepbin+1)
		pstr = "_phi"
		bins = [rbins,erbins,epbins]	
	else:
		pstr = ""
		bins = [rbins,erbins]	
	## ------------
	
	## Particles	
	Nparticles = Npbin
		
	## Initial noise drawn from Gaussian
	if a == 0.0:
		eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0, [Nparticles,2])
	else:
		eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0, [Nparticles,2])
	
	## Apply boundary conditions (should be integrated into force?)
	fxy = lambda xy, r: force(xy,r)

	## ----------------------------------------------------------------
	## Integration algorithm
	xy_step = lambda xy, r, exy: eul(xy, r, fxy, exy, dt)
	
	## ----------------------------------------------------------------

	## Filename; directory and file existence; readme
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+\
			"_INCIR_"+fstr+"_dt"+str(dt)+pstr+"/"
	hisfile = "BHIS_INCIR_"+fstr+"_a"+str(a)+"_S"+str(S)+"_dt"+str(dt)
	binfile = "BHISBIN"+hisfile[4:]
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
	
	## Save bins
	if ephi: np.savez(hisdir+binfile,rbins=rbins,erbins=erbins,epbins=epbins)
	else:	 np.savez(hisdir+binfile,rbins=rbins,erbins=erbins)

	## ----------------------------------------------------------------
	## SIMULATION
	
	if vb: print me+"Computing",Nparticles,"trajectories. Arena: CIR. Force: "+str(ftype)+"."
	
	## Precompute exp(-t)
	if a == 0:
		## Not used in calculations
		expmt = None
	elif a <= 0.1:
		raise ValueError, me+"Must have alpha>0.1"
	else:
		## a is large enough that the exponential is well resolved.
		expmt = np.exp((np.arange(-10*a,dt,dt))/a)
	
	simulate_trajectory = lambda xyini, eIC, vb2:\
							boundary_sim(xyini, eIC, a, xy_step, rmax, dt, tmax, expmt, ephi, vb2)
		
	## ----------------------------------------------------------------
	
	## Initialise histogram in space
	H = np.zeros([b.size-1 for b in bins])
	
	## Counter for noise initial conditions
	i = 0

	## Loop over initial coordinates
	for pini in pbins:
		## Perform several runs in Cartesian coordinates
		xyini = [rini*np.cos(pini),rini*np.sin(pini)]
		if vb: print me+"Run",i,"of",Nparticles
		coords = simulate_trajectory(xyini, eIC[i], vb)
		H += np.histogramdd(coords,bins=bins,normed=False)[0]
		i += 1
	## Divide by bin area and number of particles
	binc = [np.diff(b) for b in bins]
 	H /= reduce(np.multiply, np.ix_(*binc))	
	H /= Nparticles
	
	check_path(filepath, vb)
	save_data(filepath, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return
	
## ====================================================================

def boundary_sim(xyini, exyini, a, xy_step, rmax, dt, tmax, expmt, ephi, vb):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	"""
	me = "LE_inSBS.boundary_sim: "
		
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
	j = 0; sign = +1.0
	for i in xrange(0,nstp-1):
		r = np.sqrt((xy[i]*xy[i]).sum())
		if r>rmax:
			sign*=-1.0	## BC: eta sign change bounces particle back -- okay because of symmetry
			j += 1
		xy[i+1] = xy[i] + xy_step(xy[i],r,sign*exy[i])
		
	if vb: print me+"Simulation of x",round(time.time()-t0,2),"seconds for",nstp,"steps"
	if (vb and j>0): print me+"Particle reflected",j,"times"
	
	rcoord = np.sqrt((xy*xy).sum(axis=1))
	ercoord = np.sqrt((exy*exy).sum(axis=1))
	
	if ephi:
		epcoord = np.arctan2(exy[:,1],exy[:,0])
		return [rcoord, ercoord, epcoord]
	else:
		return [rcoord, ercoord]
	
## ====================================================================

def eul(xy, r, fxy, exy, dt):
	"""
	Euler step.
	Basic routine with all dependencies.
	"""
	return dt * ( fxy(xy,r) + exy )

## ====================================================================
## FORCES
	
def force_linin(xy,r,S):
	return +(S-r)*xy/r * (r<S)
	
## ====================================================================
## ====================================================================
if __name__=="__main__": input()
