import numpy as np
from scipy.signal import fftconvolve
import optparse, os, time
from datetime import datetime

from LE_Utils import save_data
from LE_LightBoundarySim import check_path, create_readme, sim_eta

def input():
	"""
	PURPOSE
		Simulate coloured noise trajectories in 2D disc geometry.
		
	INPUT
		-a --alpha		0.1		Slope of the potential
		-R --bulkrad	10.0	Position of wall
		   --HO			False	Switch from linear to harmonic potential
		-r --nruns		100		Number of runs for each (x0,y0)
		   --dt			0.01	Timestep
		-t --time		1.0		Multiply t_max by factor
	
	FLAGS
		-v --verbose	False	Print useful information to screen
		-h --help		False	Print docstring and exit
	"""	
	me = "LE_SBS.input: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## INPUT OPTIONS
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
        dest="a",default=0.1,type="float")
	parser.add_option('-R','--outrad',
        dest="R",default=10.0,type="float")
	parser.add_option('-S','--inrad',
        dest="S",default=-1.0,type="float")
	parser.add_option("-f","--ftype",
		dest="ftype",default="const",type="str")
	parser.add_option('-r','--nrun',
        dest="Nrun",default=1,type="int")
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
	R		= opts.R
	S		= opts.S
	ftype	= opts.ftype
	Nrun	= opts.Nrun
	dt		= opts.dt
	timefac = opts.timefac
	vb		= opts.vb
	
	if ftype[0] == "d":
		assert S>=0.0, me+"Must specify inner radius S for double circus."
	fpar = [R,S]
			
	if vb: print "\n==\n"+me+"Input parameters:\n\t",opts
	
	main(a,ftype,fpar,Nrun,dt,timefac,vb)
	
	return

##=============================================================================

def main(a,ftype,fpar,Nrun,dt,timefac,vb):
	"""
	"""
	me = "LE_SBS.main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## CHOOSE FORCE
	R, S = fpar[:2]
	if ftype == "const":
		force = lambda xy, r, r2: force_const(xy,r,r2,R,R*R)
		fstr = "C"
		fparstr = ""
	elif ftype == "lin":
		force = lambda xy, r, r2: force_lin(xy,r,r2,R,R*R)
		fstr = "L"
		fparstr = ""
	elif ftype == "lico":
		force = lambda xy, r, r2: force_lico(xy,r,r2,R,R*R,g)
		fstr = "LC"
		fparstr = ""
	elif ftype == "dcon":
		force = lambda xy, r, r2: force_dcon(xy,r,r2,R,R*R,S,S*S)
		fstr = "DC"
		fparstr = "_S"+str(S)
	elif ftype == "dlin":
		force = lambda xy, r, r2: force_dlin(xy,r,r2,R,R*R,S,S*S)
		fstr = "DL"
		fparstr = "_S"+str(S)
	elif ftype == "tan":
		force = lambda xy, r, r2: force_tan(xy,r,r2,R,R*R)
		fstr = "T"
		fparstr = ""
	elif ftype == "dtan":
		force = lambda xy, r, r2: force_dtan(xy,r,r2,R,R*R,S,S*S)
		fstr = "DT"
		fparstr = "_S"+str(S)
	else:
		raise IOError, me+"ftype must be one of {const, lin, lico, dcon, dlin, tan, dtan}."
	
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS
	
	## Simulation time
	tmax = 5e2*timefac
	
	## Simulation limits
	rmax = R+4.0 if ftype[-3:] is not "tan" else R+1.0
	rmin = 0.0#max([0.0, 0.9*R-5*np.sqrt(a)])
	## Injection x coordinate
	rini = 0.5*(S+R) if ftype[0] is "d" else 0.5*(rmin+R)
		
	## Histogramming; bin edges
	Nrbin = int(150 * rmax)	## Ensures number of bins per unit length
	Npbin = 50
	rbins = np.linspace(rmin,rmax,Nrbin+1)
	pbins = np.linspace(0.0,2*np.pi,Npbin)
	ermax = 4/np.sqrt(a) if a!=0 else 4/np.sqrt(dt)
	Nerbin = 150
	erbins = np.linspace(0.0,ermax,Nerbin+1)
	bins = [rbins,erbins]
	
	## Particles	
	Nparticles = Npbin*Nrun

	## Initial noise drawn from Gaussian
	if a > 0.0:
		eIC = np.random.normal(0.0,1.0/np.sqrt(a),[Nparticles,2])
	else:
		eIC = 10*(np.random.random([Nparticles,2])-0.5)
		
	## ----------------------------------------------------------------

	## Filename; directory and file existence; readme
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+\
			"_CIR_"+fstr+"_dt"+str(dt)+"/"
	hisfile = "BHIS_CIR_"+fstr+"_a"+str(a)+"_R"+str(R)+fparstr+"_dt"+str(dt)
	binfile = "BHISBIN"+hisfile[4:]
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
	
	## Save bins
	np.savez(hisdir+binfile,rbins=rbins,erbins=erbins)
	
	## ----------------------------------------------------------------
	## SIMULATION
	
	if vb: print me+"Computing",Nparticles,"trajectories. Arena: CIR. Force: "+str(ftype)+"."
	
	## ----------------------------------------------------------------
	
	## Precompute exp(-t)
	if a == 0:
		## Not used in calculations
		expmt = None
	elif a <= 0.1:
		## a is small, and exponential is dominated by first term
		if vb: print me+"rescaling time"
		expmt = np.exp(np.arange(-10,0.1,0.1))
	else:
		## a is large enough that the exponential is well resolved.
		expmt = np.exp((np.arange(-10*a,dt,dt))/a)
		
	## ----------------------------------------------------------------
	
	
	## Initialise histogram in space
	# H = np.zeros((Nrbin-1,Nerbin-1))
	H = np.zeros((Nrbin,Nerbin))
	## Counter for noise initial conditions
	i = 0

	## Loop over initial coordinates
	for pini in pbins:
		## Perform several runs in Cartesian coordinates
		xyini = [rini*np.cos(pini),rini*np.sin(pini)]
		for run in xrange(Nrun):
			if vb: print me+"Run",i,"of",Nparticles
			## r, er are radial coordinates as a function of time
			r,er = boundary_sim(xyini, eIC[i], a, force, fpar, rmin, rmax, dt, tmax, expmt, (vb and run%50==0))
			H += np.histogram2d(r,er,bins=bins,normed=False)[0]
			i += 1
	## Divide by bin area and number of particles
	H /= np.outer(np.diff(rbins),np.diff(erbins))
	H /= Nparticles
	
	check_path(filepath, vb)
	save_data(filepath, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return
	
## ====================================================================

def boundary_sim(xyini, exyini, a, force, fpar, rmin, rmax, dt, tmax, expmt, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	Dynamically adds more space to arrays.
	"""
	me = "LE_SBS.boundary_sim: "
	
	rmin2 = rmin*rmin
	
	## Initialisation
	x0,y0 = xyini
	r2 = x0*x0+y0*y0
	nstp = int(tmax/dt)
	exstp = nstp/10
	
	## Simulate eta
	if vb: t0 = time.time()
	exy = np.vstack([sim_eta(exyini[0], expmt, nstp, a, dt), sim_eta(exyini[1], expmt, nstp, a, dt)])
	if vb: print me+"Simulation of eta",round(time.time()-t0,2),"seconds for",nstp,"steps"
		
	## Spatial variables
	if vb: t0 = time.time()
		
	xy = np.zeros([2,nstp]); xy[:,0] = [x0,y0]
	j = 0
	## Calculate trajectory
	for i in xrange(0,nstp-1):
		r2 = (xy[:,i]*xy[:,i]).sum()
		fxy = force(xy[:,i],np.sqrt(r2),r2)
		xy[:,i+1] = xy[:,i] + dt*( fxy + exy[:,i] )
		## Apply BC
		if r2 < rmin2:
			xy[:,i] *= 1-(2*rmin)/np.sqrt(r2)
			j += 1
	if (vb and j==0 and rmin2>0.0): print me+"rmin never encountered."
	if vb: print me+"Simulation of x",round(time.time()-t0,2),"seconds for",nstp,"steps"
	
	# xx = np.linspace(0,2*np.pi,100)
	# plt.plot(fpar[0]*np.cos(xx),fpar[0]*np.sin(xx),fpar[1]*np.cos(xx),fpar[1]*np.sin(xx))
	# plt.plot(*xy);plt.show();exit()
	
	rcoord = np.sqrt((xy*xy).sum(axis=0))
	ercoord = np.sqrt((exy*exy).sum(axis=0))
	
	return rcoord, ercoord
	
## ====================================================================

def force_const(xy,r,r2,R,R_2):
	return 0.5*(np.sign(R_2-r2)-1) * xy/(r+0.0001*(r==0.0))

def force_lin(xy,r,r2,R,R_2):
	return force_const(xy,r,r2,R,R_2) * (r-R)
	
def force_lico(xy,r,r2,R,R_2,g):
	"""NOT TESTED"""
	return force_lin(xy,r,r2,R,R2) + g
	
def force_dcon(xy,r,r2,R1,R1_2,R2,R2_2):
	return force_const(xy,r,r2,R1,R1_2) - force_const(xy,r,-r2,R2,-R2_2)
	
def force_dlin(xy,r,r2,R1,R1_2,R2,R2_2):
	return force_lin(xy,r,r2,R1,R1_2) + force_lin(xy,r,-r2,R2,-R2_2)

def force_tan(xy,r,r2,R,R_2):
	return force_const(xy,r,r2,R,R_2) * 0.5*np.pi*np.tan(0.5*np.pi * (r-R)/1.0)
	
def force_dtan(xy,r,r2,R1,R1_2,R2,R2_2):
	return force_tan(xy,r,r2,R1,R1_2) + force_tan(xy,r,-r2,R2,-R2_2)

## ====================================================================
## ====================================================================
if __name__=="__main__": input()
