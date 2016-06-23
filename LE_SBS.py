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
		Simulate coloured noise trajectories in 2D disc / annular geometry.
		
	INPUT
		-a	--alpha		0.1		Slope of the potential
		-R	--outrad	2.0		Position of outer wall
		-S	--inrad		-1.0	Position of inner wall
			--lam		-1.0	Width of walls for infinite potentials
			--nu		-1.0	Potential multiplier for log-quadratic potential
			--ftype		const	const, lin, tan, nu, dcon, dlin, dtan, dnu
		-r 	--nruns		1		Number of runs for each (x0,y0)
		   	--dt		0.01	Timestep
		-t 	--timefac	1.0		Multiply t_max by factor
			--intmeth	euler	Integration method. euler, rk2, rk4
	
	FLAGS
			--ephi		False	Histogram in eta-phi also, for bulk-constant plots
		-v	--verbose	False	Print useful information to screen
		-h	--help		False	Print docstring and exit
		
	TODO
		Consistent passing r2 or r to force functions
		Calculate and save r as we go rather than duplicating effort
	"""	
	me = "LE_SBS.input: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## INPUT OPTIONS
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
		dest="a",default=0.2,type="float")
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
	parser.add_option("--intmeth",
		dest="intmeth",default="euler",type="str")
	parser.add_option("--ephi",
		dest="ephi", default=False, action="store_true")
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
	ftype	= (opts.ftype).lower()
	Nrun	= opts.Nrun
	dt		= opts.dt
	timefac = opts.timefac
	intmeth = (opts.intmeth).lower()
	ephi	= opts.ephi
	vb		= opts.vb
	
	if ftype[0] == "d":		assert S>=0.0, me+"Must specify inner radius S for double circus."
	if ftype[-3:] == "tan":	assert lam>0.0,	me+"Must specify lengthscale lambda for tan potential."
	if ftype[-2:] == "nu":	assert nu>0.0,	me+"Must specify potential multipier nu for nu potential."
	
	fpar = [R,S,lam,nu]
			
	if vb: print "\n==\n"+me+"Input parameters:\n\t",opts
	
	main(a,ftype,fpar,Nrun,dt,timefac,intmeth,ephi,vb)
	
	return

##=============================================================================

def main(a,ftype,fpar,Nrun,dt,timefac,intmeth,ephi,vb):
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
	rmax = R+2.0*lam if infpot else R+4.0
	# rmin = 0.0 #max([0.0, 0.9*R-5*np.sqrt(a)])
	rmin = max(0.0, S-2.0*lam if infpot else S-4.0)
	## Injection x coordinate
	rini = 0.5*(S+R) if dblpot else 0.5*(rmin+R)
	if R==0.0: rini += 0.001
	## Limits for finite potential
	wb = [R+lam, (S>0.0)*(S-lam)] if infpot else [False, False]
	
	## ------------
	## Bin edges
	
	rbins = calc_rbins(infpot,fpar,rmin,rmax)
	Nrbin = rbins.size - 1
	
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
	Nparticles = Npbin*Nrun

		
	## Initial noise drawn from Gaussian
	if a == 0.0:
		eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0, [Nparticles,2])
	else:
		eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0, [Nparticles,2])
	
	## Apply boundary conditions (should be integrated into force?)
	fxy = lambda xy, r: fxy_infpot(xy,r,force,wb[0],wb[1],dt) if infpot else force(xy,r)

	## Integration algorithm
	eul_step = lambda xy, r, exy: eul(xy, r, fxy, exy, dt)
	
	if intmeth == "rk4":
		xy_step = lambda xy, r, exy: RK4(xy, r, fxy, exy, dt, eul_step)
		intmeth = "_rk4"
	elif intmeth == "rk2":
		xy_step = lambda xy, r, exy: RK2(xy, r, fxy, exy, dt, eul_step)
		intmeth = "_rk2"
	else:
		xy_step = eul_step
		intmeth = ""
					
	## ----------------------------------------------------------------

	## Filename; directory and file existence; readme
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+\
			"_CIR_"+fstr+"_dt"+str(dt)+intmeth+pstr+"/"
	hisfile = "BHIS_CIR_"+fstr+"_a"+str(a)+"_R"+str(R)+fparstr+"_dt"+str(dt)+intmeth
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
	# elif a <= 0.1:
		# ## a is small, and exponential is dominated by first term
		# if vb: print me+"rescaling time"
		# expmt = np.exp(np.arange(-10,0.1,0.1))
	else:
		## a is large enough that the exponential is well resolved.
		expmt = np.exp((np.arange(-10*a,dt,dt))/a)
	
	simulate_trajectory = lambda xyini, eIC, vb2:\
							boundary_sim(xyini, eIC, a, xy_step, dt, tmax, expmt, ephi, vb2)
		
	## ----------------------------------------------------------------
	
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
			coords = simulate_trajectory(xyini, eIC[i], (vb and run%50==0))
			#t2 = time.time()
			H += np.histogramdd(coords,bins=bins,normed=False)[0]
			#if (vb and run%1==0): print me+"Histogram:",round(time.time()-t2,1),"seconds."
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

def boundary_sim(xyini, exyini, a, xy_step, dt, tmax, expmt, ephi, vb):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	"""
	me = "LE_SBS.boundary_sim: "
		
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
		r = np.sqrt((xy[i]*xy[i]).sum())
		xy[i+1] = xy[i] + xy_step(xy[i],r,exy[i])
		
	if vb: print me+"Simulation of x",round(time.time()-t0,2),"seconds for",nstp,"steps"
			
	rcoord = np.sqrt((xy*xy).sum(axis=1))
	ercoord = np.sqrt((exy*exy).sum(axis=1))
	
	## -----------------===================-----------------
	R = 2.0; S = 1.0; lam = 0.5; nu = 1.0
	## Distribution of spatial steps and eta
	if 0:
		from LE_RunPlot import plot_step_wall, plot_eta_wall, plot_step_bulk
		## In wall regions
		plot_step_wall(xy,rcoord,R,S,a,dt,vb)
		plot_eta_wall(xy,rcoord,exy,ercoord,R,S,a,dt,vb)
		## In bulk region
		plot_step_bulk(xy,rcoord,ercoord,R,S,a,dt,vb)
		exit()
	## Trajectory plot with force arrows
	if 0:
		from LE_RunPlot import plot_traj	
		plot_traj(xy,rcoord,R,S,lam,nu,force_dnu,a,dt,vb)
		exit()
	## -----------------===================-----------------
			
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
	
def RK2_new(xy, r, fxy, exy, dt, eul_step):
	"""
	RK2 (midpoint method) step. 
	Basic routine with all dependencies.
	Following Wikipedia.
	"""
	K1 = eul_step(xy, np.sqrt(r), exy)
	K2 = eul_step(xy+K1, np.sqrt(((xy+K1)*(xy+K1)).sum()), exy)
	return 0.5*(K1+K2)

def RK2(xy1, r, fxy, exy, dt, eul_step):
	"""
	RK2 (midpoint method) step. 
	Basic routine with all dependencies.
	"""
	xy2 = xy1+0.5*eul_step(xy1, r, exy)
	return eul_step(xy2, (xy2*xy2).sum(), exy)
	
def RK4(xy1, r, fxy, exy, dt, eul_step):
	"""
	RK4 step. 
	Basic routine with all dependencies.
	"""
	k1 = eul_step(xy1, r, exy)
	xy2 = xy1+0.5*k1
	k2 = eul_step(xy2, (xy2*xy2).sum(), exy)
	xy3 = xy1+0.5*k2
	k3 = eul_step(xy3, (xy3*xy3).sum(), exy)
	xy4 = xy1+k3
	k4 = eul_step(xy4, (xy4*xy4).sum(), exy)
	return 1.0/6.0 * ( k1 + 2*k2 + 2*k3 + k4 )

## ====================================================================
## FORCES
	
def fxy_infpot(xy,r,force,wob,wib,dt):
	"""
	Force for infinite potential: checking whether boundary is crossed.
	"""
	if   (wob and r>wob):	fxy = -xy/r/dt*(r-wob+(wob-wib)*np.random.rand())
	elif (wib and r<wib): 	fxy = +xy/r/dt*(wib-r+(wob-wib)*np.random.rand())
	else:					fxy = force(xy,r)
	return fxy
	
## ----------------------------------------------------------------------------

def force_const(xy,r,R):
	return -xy/(r+0.0001*(r==0.0)) * (r>R)

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
	return -lam*nu*2.0*Dr/(lam*lam-Dr*Dr+0.0001)*xy/r * (r>R)
	
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
		NrRbin = int(max(50,150*(lam)))
		NrSbin = int(max(50,150*(lam)))*(S>0.0)
		## Bulk bins
		NrBbin = int(150 * (R-S))
		rbins = np.hstack([\
				np.linspace(rmin,max(0.0,S-lam),10),\
				np.linspace(S-lam,S,NrSbin+1),\
				np.linspace(S,R,NrBbin+1),\
				np.linspace(R,R+lam,NrRbin+1),\
				np.linspace(R+lam,rmax,10)])
		rbins = np.unique(rbins)
	## Keep things simple when potential is low-order
	else:
		Nrbin = int(150 * (rmax-rmin))
		rbins = np.linspace(rmin,rmax,Nrbin+1)
	return rbins
	

## ====================================================================
## ====================================================================
if __name__=="__main__": input()
