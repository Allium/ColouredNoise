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
	parser.add_option('-l','--lambda',
		dest="lam",default=-1.0,type="float")
	parser.add_option('-n','--nu',
		dest="nu",default=-1.0,type="float")
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
	lam		= opts.lam
	nu		= opts.nu
	ftype	= opts.ftype
	Nrun	= opts.Nrun
	dt		= opts.dt
	timefac = opts.timefac
	vb		= opts.vb
	
	if ftype[0] == "d":		assert S>=0.0, me+"Must specify inner radius S for double circus."
	if ftype[-3:] == "tan":	assert lam>0.0,	me+"Must specify lengthscale lambda for tan potential."
	if ftype[-2:] == "nu":	assert nu>0.0,	me+"Must specify potential multipier nu for nu potential."
	
	fpar = [R,S,lam,nu]
			
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
	R, S, lam, nu = fpar
	if ftype == "const":
		force = lambda xy, r: force_const(xy,r,R)
		fstr = "C"
		fparstr = ""
	elif ftype == "lin":
		force = lambda xy, r: force_lin(xy,r,R)
		fstr = "L"
		fparstr = ""
	elif ftype == "lico":
		force = lambda xy, r: force_lico(xy,r,R,g)
		fstr = "LC"
		fparstr = ""
	elif ftype == "dcon":
		force = lambda xy, r: force_dcon(xy,r,R,S)
		fstr = "DC"
		fparstr = "_S"+str(S)
	elif ftype == "dlin":
		force = lambda xy, r: force_dlin(xy,r,R,S)
		fstr = "DL"
		fparstr = "_S"+str(S)
	elif ftype == "tan":
		force = lambda xy, r: force_tan(xy,r,R,lam)
		fstr = "T"
		fparstr = "_l"+str(lam)
	elif ftype == "dtan":
		force = lambda xy, r: force_dtan(xy,r,R,S,lam)
		fstr = "DT"
		fparstr = "_S"+str(S)+"_l"+str(lam)
	elif ftype == "nu":
		force = lambda xy, r: force_nu(xy,r,R,lam,nu)
		fstr = "N"
		fparstr = "_l"+str(lam)+"_n"+str(nu)
	elif ftype == "dnu":
		force = lambda xy, r: force_dnu(xy,r,R,S,lam,nu)
		fstr = "DN"
		fparstr = "_S"+str(S)+"_l"+str(lam)+"_n"+str(nu)
	else:
		raise IOError, me+"ftype must be one of {const, lin, lico, dcon, dlin, tan, dtan, nu}."
	
	dblpot = True if ftype[0] == "d" else False
	infpot = True if (ftype[-3:] == "tan" or ftype[-2:] == "nu") else False
	
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS
	
	## Simulation time
	tmax = 5e2*timefac
	
	## Simulation limits
	rmax = R+lam if infpot else R+5.0
	rmin = 0.0 #max([0.0, 0.9*R-5*np.sqrt(a)])
	## Injection x coordinate
	rini = 0.5*(S+R) if dblpot else 0.5*(rmin+R)
	## Limits for finite potential
	wb2 = [(R+lam)**2,(S>0.0)*(S-lam)**2] if infpot else [False,False]
	
	## ------------
	## Bin edges
	
	rbins = calc_rbins(infpot,fpar,rmin,rmax)
	Nrbin = rbins.size - 1
	
	Npbin = 50
	pbins = np.linspace(0.0,2*np.pi,Npbin)
	
	ermax = 4/np.sqrt(a) if a!=0 else 4/np.sqrt(dt)
	Nerbin = 150
	erbins = np.linspace(0.0,ermax,Nerbin+1)
	
	Nepbin = 50
	epbins = np.linspace(0.0,2*np.pi,Nepbin+1)
	
	bins = [rbins,erbins,epbins]	
	## ------------
	
	## Particles	
	Nparticles = Npbin*Nrun

	
	########## CHANGE (new definitions)
	fxy = lambda xy, r2: fxy_infpot(xy,r2,force,wob2,wib2) if infpot else fxy_finpot(xy,r2,force)
	
	## Initial noise drawn from Gaussian
	if a == 0.0:
		eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0, [Nparticles,2])
	else:
		eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0, [Nparticles,2])
		
	eul_step = lambda xy, r2, exy: eul(xy, r2, fxy, exy, dt)
	RK4_step = lambda xy, r2, exy: RK4(xy, r2, fxy, exy, dt, eul_step)
	xy_step = RK4_step if a==0.0 else eul_step
	##########
		
	## ----------------------------------------------------------------

	## Filename; directory and file existence; readme
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+\
			"_CIR_"+fstr+"_dt"+str(dt)+"_phi/"
	hisfile = "BHIS_CIR_"+fstr+"_a"+str(a)+"_R"+str(R)+fparstr+"_dt"+str(dt)
	binfile = "BHISBIN"+hisfile[4:]
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
	
	## Save bins
	np.savez(hisdir+binfile,rbins=rbins,erbins=erbins,epbins=epbins)
	
	## ----------------------------------------------------------------
	## SIMULATION
	
	if vb: print me+"Computing",Nparticles,"trajectories. Arena: CIR. Force: "+str(ftype)+"."
	
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
	
	########## CHANGE (lambdify boundary_sim)
	simulate_trajectory = lambda xyini, eIC, vb2: boundary_sim(xyini, eIC, a, xy_step, rmin, dt, tmax, expmt, vb2)
	##########
	
	## Initialise histogram in space
	H = np.zeros([b.size-1 for b in bins])
	## Counter for noise initial conditions
	i = 0

	## Loop over initial coordinates
	for pini in pbins:
		## Perform several runs in Cartesian coordinates
		xyini = [rini*np.cos(pini),rini*np.sin(pini)]
		for run in xrange(Nrun):
			if vb: print me+"Run",i,"of",Nparticles
			########## CHANGE
			coords = simulate_trajectory(xyini, eIC[i], (vb and run%50==0))
			########## 
			t2 = time.time()
			H += np.histogramdd(coords,bins=bins,normed=False)[0]
			if (vb and run%1==0): print me+"Histogram:",round(time.time()-t2,1),"seconds."
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

def boundary_sim(xyini, exyini, a, xy_step, rmin, dt, tmax, expmt, vb):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	"""
	me = "LE_SBS.boundary_sim: "
	
	rmin2 = rmin*rmin
	#wob2, wib2 = wb2
	
	## Initialisation
	x0,y0 = xyini
	r2 = x0*x0+y0*y0
	nstp = int(tmax/dt)
	exstp = nstp/10
	
	## Simulate eta
	if vb: t0 = time.time()
	########## CHANGE (order of dimensions)
	exy = np.vstack([sim_eta(exyini[0], expmt, nstp, a, dt), sim_eta(exyini[1], expmt, nstp, a, dt)]).T
	##########
	if vb: print me+"Simulation of eta",round(time.time()-t0,2),"seconds for",nstp,"steps"
		
	## Spatial variables
	if vb: t0 = time.time()
		
	########## CHANGE (order of dimensions)
	xy = np.zeros([nstp,2]); xy[0] = [x0,y0]
	j = 0
	## Calculate trajectory
	for i in xrange(0,nstp-1):
		r2 = (xy[i]*xy[i]).sum()
		xy[i+1] = xy[i] + xy_step(xy[i],r2,exy[i])
	##########
		
	if vb: print me+"Simulation of x",round(time.time()-t0,2),"seconds for",nstp,"steps"
			
	########## CHANGE (order of dimensions)
	rcoord = np.sqrt((xy*xy).sum(axis=1))
	ercoord = np.sqrt((exy*exy).sum(axis=1))
	epcoord = np.arctan2(exy[:,1],exy[:,0])
	##########
	
	return [rcoord, ercoord, epcoord]
	
## ====================================================================


def fxy_finpot(xy,r2,force):
	"""
	Force for finite potential.
	"""
	return force(xy,np.sqrt(r2))
	
def fxy_infpot(xy,r2,force,wob2,wib2):
	"""
	Force for infinite potential: must check whether boundary is crossed.
	"""
	if   (wob2 and r2>wob2):	fxy = -1e10*xy/np.sqrt(r2)
	elif (wib2 and r2<wib2): 	fxy = +1e10*xy/np.sqrt(r2)
	else:						fxy = force(xy,np.sqrt(r2))
	return fxy
	
## ----------------------------------------------------------------------------

def eul(xy, r2, fxy, exy, dt):
	"""
	Euler step.
	Basic routine with all dependencies.
	"""
	return dt * ( fxy(xy,r2) + exy )

def RK4(xy1, r2, fxy, exy, dt, eul_step):
	"""
	RK4 step. 
	Basic routine with all dependencies.
	Only appropriate for white noise.
	"""
	k1 = eul_step(xy1, r2, exy)
	xy2 = xy1+0.5*k1
	k2 = eul_step(xy2, (xy2*xy2).sum(), exy)
	xy3 = xy1+0.5*k2
	k3 = eul_step(xy3, (xy3*xy3).sum(), exy)
	xy4 = xy1+k3
	k4 = eul_step(xy4, (xy4*xy4).sum(), exy)
	return 1.0/6.0 * ( k1 + 2*k2 + 2*k3 + k4 )

## ====================================================================

def force_const(xy,r,R):
	return -xy/r * (r>R)

def force_lin(xy,r,R):
	return -(r-R)*xy/r * (r>R)
	
def force_dcon(xy,r,R,S):
	return force_const(xy,r,R) + force_const(xy,-r,-S)
	
def force_dlin(xy,r,R,S):
	return force_lin(xy,r,R) + force_lin(xy,-r,-S)

## Finite-distance
	
def force_tan(xy,r,R,lam):
	return -0.5*np.pi*np.tan(0.5*np.pi*(r-R)/lam)*xy/r * (r>R)
	
def force_nu(xy,r,R,lam,nu):
	Dr = (r-R) * (r>R)
	return -lam*nu*2.0*Dr/(lam*lam-Dr*Dr)*xy/r * (r>R)
	
def force_dtan(xy,r,R,S,lam):
	return force_tan(xy,r,R,lam) + force_tan(xy,-r,-S,lam)
	
def force_dnu(xy,r,R,S,lam,nu):
	return force_nu(xy,r,R,lam,nu) + force_nu(xy,-r,-S,lam,nu)
	
## ====================================================================

def calc_rbins(finipot, fpar, rmin, rmax):
	"""
	Want to ensure a sensible number of bins per unit length.
	Returns positins of bin edges.
	"""
	## When potential changes rapidly, need many bins
	if finipot:
		R, S, lam = fpar[:3]
		## Wall bins
		NrRbin = int(max(50,150*(rmax-R)))
		NrSbin = int(max(50,150*(S-rmin)))*(S>0.0)
		## Bulk bins
		NrBbin = int(150 * (R-S))
		rbins = np.hstack([np.linspace(rmin,S,NrSbin+1),\
				np.linspace(S,R,NrBbin+1),\
				np.linspace(R,rmax,NrRbin+1)])
		rbins = np.unique(rbins)
	## Keep things simple when potential is low-order
	else:
		Nrbin = int(150 * (rmax-rmin))
		rbins = np.linspace(rmin,rmax,Nrbin+1)
	return rbins
	

## ====================================================================
## ====================================================================
if __name__=="__main__": input()
