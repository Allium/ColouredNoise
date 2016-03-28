import numpy as np
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import optparse
from scipy.signal import fftconvolve

from LE_Utils import save_data

from LE_LightBoundarySim import check_path, create_readme, sim_eta

def input():
	"""
	NAME
		LE_SBS.py
	
	PURPOSE
		Simulate coloured noise trajectories in 2D circular geometry.
	
	EXECUTION
		
	FLAGS
		-a --alpha		0.1		Slope of the potential
		-R --bulkrad	10.0	Position of wall
		   --HO			False	Switch from linear to harmonic potential
		-r --nruns		100		Number of runs for each (x0,y0)
		   --dt			0.01	Timestep
		-t --time		1.0		Multiply t_max by factor
		-v --verbose	False	Print useful information to screen
		-h --help		False	Print docstring and exit
	
	EXAMPLE
	
	NOTES
	
	BUGS
	
	HISTORY
		10 March 2016	Started
	"""	
	me = "LE_SBS.input: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## INPUT OPTIONS
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
        dest="a",default=0.1,type="float")
	parser.add_option('-R','--bulkrad',
        dest="R",default=10.0,type="float")
	parser.add_option("--HO",
		dest="harmonic_potential",default=False,action="store_true")
	parser.add_option('-r','--nrun',
        dest="Nrun",default=100,type="int")
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
	Nrun	= opts.Nrun
	dt		= opts.dt
	timefac = opts.timefac
	vb		= opts.vb
	
	## Choose potential type
	force = force_lin if opts.harmonic_potential else force_const
		
	if vb: print "\n==\n"+me+"Input parameters:\n\t",opts
	
	main(a,R,force,Nrun,dt,timefac,vb)
	
	return

##=============================================================================

def main(a,R,force,Nrun,dt,timefac,vb):
	"""
	"""
	me = "LE_SBS.main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS
	
	## Simulation time
	tmax = 5e2*timefac
	
	R2 = R*R
	## Simulation limits
	rmax = R+5.0
	# rmin = 0.9*R - (max(5*np.sqrt(a),2*dt/a) if a!=0.0 else np.sqrt(2))
	rmin = 0.9*R - (5*np.sqrt(a) if a!=0.0 else np.sqrt(2))
	rmin = max(0.0,rmin)
	## Injection x coordinate
	rini = 0.5*(rmin+R)
		
	## Histogramming; bin edges
	Nrbin = 200
	Npbin = 50
	rbins = np.linspace(rini,rmax,Nrbin)
	pbins = np.linspace(0.0,2*np.pi,Npbin)
	
	## Particles	
	Nparticles = pbins.size*Nrun

	## Initial noise drawn from Gaussian
	if a > 0.0:
		eIC = np.random.normal(0.0,1.0/np.sqrt(a),[Nparticles,2])
	else:
		eIC = 100*(np.random.random([Nparticles,2])-0.5)
		
	## ----------------------------------------------------------------

	## Filename; directory and file existence; readme
	f_type = "C" if force == force_const else "L"
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+\
			"_CIR_"+f_type+"_r"+str(Nrun)+"_dt"+str(dt)+"/"
	hisfile = "BHIS_CIR_"+f_type+"_a"+str(a)+"_R"+str(R)+"_r"+str(Nrun)+"_dt"+str(dt)
	binfile = "BHISBIN"+hisfile[4:]
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
	
	## Save bins
	np.savez(hisdir+binfile,rbins=rbins,pbins=pbins)
			
	## ----------------------------------------------------------------
	## SIMULATION
	
	if vb: print me+"Computing",Nparticles,"trajectories."
	
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
	H = np.zeros((Nrbin-1,Npbin-1))
	## Counter for noise initial conditions
	i = 0
	
	## Loop over initial coordinates
	for pini in pbins:
		## Perform several runs in Cartesian coordinates
		xyini = [rini*np.cos(pini),rini*np.sin(pini)]
		for run in xrange(Nrun):
			## x, y are coordinates as a function of time
			x, y = boundary_sim(xyini, eIC[i], a, R, force, rmin, rmax, dt, tmax, expmt, (vb and run%50==0))
			# plot_traj(np.sqrt(x*x+y*y),np.arctan2(y,x),rmin,R,rmax,hisdir+"TRAJ"+hisfile[4:]+".png")
			H += np.histogram2d(x,y,bins=[rbins,pbins],normed=False)[0]
			i += 1
	## Divide by bin area and number of particles
	H /= np.outer(np.diff(rbins),np.diff(pbins))
	# H /= np.outer(0.5*(rbins[1:]+rbins[-1])*np.diff(rbins),np.diff(pbins))
	H /= Nparticles
	## Azimuthal average
	H = H.sum(axis=1)
	save_data(filepath, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return
	
## ====================================================================

def boundary_sim(xyini, exyini, a, R, force, rmin, rmax, dt, tmax, expmt, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	Dynamically adds more space to arrays.
	"""
	me = "LE_SBS.boundary_sim: "
	
	R2 = R*R
	rmin2 = max(rmin*rmin,0.1*0.1)
	
	## Initialisation
	x0,y0 = xyini
	r2 = x0*x0+y0*y0
	nstp = int(tmax/dt)
	exstp = nstp/10
	if vb: print me+"[a, R] =",np.around([a,R],2),"; [x0, y0] =",np.around(xyini,2)
	
	## Simulate eta
	if vb: t0 = time.time()
	exy = np.vstack([sim_eta(exyini[0], expmt, nstp, a, dt), sim_eta(exyini[1], expmt, nstp, a, dt)])
	if vb: print me+"Simulation of eta",round(time.time()-t0,2),"seconds for",nstp,"steps"
		
	## Spatial variables
	if vb: t0 = time.time()
		
	xy = np.zeros([2,nstp]); xy[:,0] = [x0,y0]
	i,j = 0,0
	## Euler steps to calculate x(t)
	while r2 > rmin2:
		r2 = (xy[:,i]*xy[:,i]).sum()
		fxy = force(xy[:,i],np.sqrt(r2),r2,R,R2)
		xy[:,i+1] = xy[:,i] + dt*( fxy + exy[:,i] )
		i +=1
		## Extend array if necessary
		if i == xy.shape[1]-1:
			exy_2 = np.vstack([sim_eta(exy[-1,0],expmt,exstp,a,dt),sim_eta(exy[-1,1],expmt,exstp,a,dt)])
			exy = np.hstack([exy,exy_2])
			xy = np.hstack([xy,np.zeros([2,exstp])])
			j += 1
	if (j>0 and vb): print me+"trajectory array extended",j,"times."
	## Clip trailing zeroes
	xy = xy[:,:i]
	if vb: print me+"Simulation of x",round(time.time()-t0,2),"seconds for",i,"steps"

	return xy
	
## ====================================================================

def force_const(xy,r,r2,R,R2):
	return 0.5*(np.sign(R2-r2)-1) * xy/r

def force_lin(xy,r,r2,R,R2):
	return force_const(xy,r,r2,R,R2) * (r-R)

## ====================================================================

def plot_traj(rad,theta,rmin,R,rmax,outfile):
	me = "LE_SBS.plot.traj: "
	
	ax = plt.subplot(111, projection="polar")
	
	## Plot wall and simulation boundary
	TH = np.linspace(-np.pi,np.pi,360)
	ax.plot(TH,R*np.ones(360),"k-",linewidth=2)
	ax.plot(TH,rmin*np.ones(360),"b-",linewidth=1)
	
	## Plot trajectory
	ax.plot(theta, rad, "r-")
	ax.set_rlim([0.0,rmax])
	ax.grid(True)
	plt.show()
	
	# plt.savefig(outfile)
	# print me+"Figure saved as",outfile
	plt.clf()
	return


## ====================================================================
## ====================================================================
if __name__=="__main__": input()
