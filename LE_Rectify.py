me0 = "LE_Rectifier"

import numpy as np
import scipy as sp
import os, glob, optparse, time
from datetime import datetime
from platform import system

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from LE_LightBoundarySim import sim_eta
from LE_CSim import eul

from LE_Utils import fs, set_mplrc, filename_par
set_mplrc(fs)


##=============================================================================
def main():
	"""
	An asymmetric periodic potential leads to net current.
	"""
	me = me0+".main: "
	t0 = time.time()

	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-a','--alpha',
		dest="alpha", type="float", default=1.0)
	parser.add_option('-L',
		dest="L", type="float", default=0.8)
	parser.add_option('-l','--lam',
		dest="lam", type="float", default=1.0)
	parser.add_option('-t','--tfac',
		dest="tfac", type="float", default=1.0)
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('--jpg',
		dest="jpg", default=False, action="store_true")
	parser.add_option('--png',
		dest="png", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	
	if opt.jpg:	fs["saveext"]="jpg"
	if opt.png:	fs["saveext"]="png"
	vb = opt.verbose
	
	a = opt.alpha
	L = opt.L
	lam = opt.lam
	tmax = 15e2*opt.tfac
	dt = 0.01

	if opt.nosave:
		outdir = False
		opt.showfig = True
	else:
		outdir = "./Pressure/RECT/RECT_CAR_L_t%.1f/"%(opt.tfac)
		if not os.path.isdir(outdir):
			os.mkdir(outdir)
			print me+"Created directory",outdir
	
	## Plot wind
	if 1:
		plot_wind(lam,tmax,dt,outdir,vb)
	
	## Plot density
	else:
		x, ex, wind = sim(a,L,lam,tmax,dt,vb)
		plot_density(x, ex, a, L, lam, wind, outdir, vb)

	if vb:	print me+"Total time %.1f seconds."%(time.time()-t0)

	if opt.showfig:	plt.show()
	
	return
	
##=================================================================================================
def sim(a,L,lam,tmax,dt,vb):
	"""
	"""
	me = me0+".sim: "
	t0 = time.time()

	## Injection x coordinate
	xini = 0.0

	## Initial noise drawn from Gaussian
	if a == 0.0:	eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0)
	else:			eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0)
	
	## Functions to pass
	f_step = lambda x: force_rect(x, L, lam)
	x_step = lambda x, ex: eul(x, ex, f_step, dt)
	
	## ----------------------------------------------------------------
	## Simulation

	## Precompute exp(-t)
	expmt = np.exp((np.arange(-10*a,dt,dt))/a)

	x, ex = boundary_sim(xini, eIC, a, x_step, dt, tmax, expmt)
		
	## Boundary conditions
	wind = int(x[-1]/lam)/tmax
	x = x%lam
	
	if vb:	print me+"Winding rate %.1g"%(wind)
	if vb:	print me+"Simulation [a,L,lam]=[%.1f,%.1f,%.1f]: %.1f seconds."%(a,L,lam,time.time()-t0)
	
	return x, ex, wind
	
##=================================================================================================
def plot_density(x, ex, a, L, lam, wind, outdir, vb):
	"""
	Given trajectory, plot the density of points.
	"""
	me = me0+".plot_density: "
	t0 = time.time()
		
	## Bins
	xbins = np.linspace(0.0,lam,200*lam+1)
	emax = 3/np.sqrt(a)
	ebins = np.linspace(-emax,+emax,100*emax+1)
	
	## Force and potential
	fx = force_rect(xbins, L, lam)
	U  = -sp.integrate.cumtrapz(fx, xbins, initial=0.0); U-=U.min()
	
	## Density
	H = np.histogramdd([x,ex],bins=[xbins,ebins],normed=True)[0]

	## ----------------------------------------------------------------
	## Plotting
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	plt.set_cmap("Greys")
	
	## Plot data
	xc = 0.5*(xbins[:-1]+xbins[1:])
	ec = 0.5*(ebins[:-1]+ebins[1:])
	ax.contourf(xc,ec,H.T)
	
	## Potential
	ax.plot(xbins, (U*0.5*emax/U.max())-emax, "k--", label=r"$U(x)$")
	
	## Figure
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	ax.set_title(r"$\alpha=%.1f$. Winding rate %.1g."%(a,wind))
	ax.grid()
		
	## ------------------------------------------------------------------------
	
	if outdir:
		plotfile = outdir+"/PDFxex_a%.1f_L%.1f_l%.1f."%(a,L,lam)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Density plot saved to",plotfile
	
	if vb:	print me+"Plotting and saving %.1f seconds."%(time.time()-t0)
		
	return

##=================================================================================================
def plot_wind(lam,tmax,dt,outdir,vb):
	"""
	Run several simulations to see how winding number changes.
	"""
	me = me0+".plot_wind: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh using phase_calc
	
	windfile = outdir+"/WIND.npz"
	
	try:
		data = np.load(windfile)
		print me+"Data file found:",windfile
		A, L, W = data["A"], data["L"], data["W"]
		del data
		
	except IOError:
		print me+"No data found. Calculating."
	
		alist = [0.2,0.5,1.0,2.0,3.0,5.0,10.0]
		Llist = lam*np.linspace(0.5,0.9,5)
		W = np.zeros([len(alist),len(Llist)])
		
		A, L = [], []
		for i,ai in enumerate(alist):
			for j,Li in enumerate(Llist):
				A += [ai]
				L += [Li]
				x, ex, W[i,j] = sim(ai,Li,lam,tmax,dt,vb)
				# plot_density(x, ex, a, L, lam, W[i,j], outdir, vb); plt.close()
		
		W = np.array(W)
		np.savez(windfile, A=A, L=L, W=W)
	
	## ------------------------------------------------------------------------
	## Plotting
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	plt.set_cmap("seismic")
	
	im = ax.scatter(A, L, c=W.flatten(), marker="o", s=100)
	
	## Keys
	cbar = fig.colorbar(im, ax=ax, aspect=50)
	cbar.ax.get_yaxis().labelpad = 30
	cbar.ax.set_ylabel(r"Winding rate", rotation=270, fontsize=fs["fsl"]-4)
	wmax = max([W.max(),-W.min()])
	cbar.set_clim(vmin=-wmax,vmax=+wmax)
	cbar.ax.tick_params(labelsize=fs["fsl"]-4)
	cbar.locator = MaxNLocator(5);	cbar.update_ticks()
	plt.subplots_adjust(right=1.0)
	
	ax.set_xlim(0.0,A.max())
	ax.set_ylim(0.49,1.0)
	
	## Figure
	ax.set_xlabel(r"$\alpha$")
	ax.set_ylabel(r"$L/\lambda$")
	ax.grid()
	
	## ------------------------------------------------------------------------
	
	if outdir:
		plotfile = outdir+"/WIND."+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Density plot saved to",plotfile
	
	if vb:	print me+"Plotting and saving %.1f seconds."%(time.time()-t0)
		
	return


##=================================================================================================
def plot_traj():
	raise NotImplementedError
	## ----------------------------------------------------------------
	## PLOT TRAJECTORY
	
	exit()
	print me+"Starting to plot trajectories."
	
	outdir = outdir+"/RECT_CAR_L_a%.1f_L%.1f_l%.1f_t%.1f"%(a,L,lam,opt.tfac)

	
	## Coarsen to speed up plotting
	crsn = 5
	if crsn>1:
		x = x[::crsn]
		ex = ex[::crsn]
	
	## Number of steps per frame
	nsteps = int(tmax/dt/crsn)
	stride = 50/crsn	## Make a frame every stride timesteps
	numf = nsteps/stride
	
	## Loop over frames
	for fnum in range(numf):
		ti = time.time()
		fig = plot_traj(x[:fnum*stride],ex[:fnum*stride],L,lam,a,dt,x.size,vb)
	
		## Save
		plotfile = outdir+"/f%04d."%(fnum)+fs["saveext"]
		fig.savefig(plotfile)
		plt.close()
		if not fnum%10: print me+"Frame %04d/%04d saved. Time per file %.1f seconds."%(fnum,numf,time.time()-ti)

	return
	
##=============================================================================
	
def plot_traj(x,ex,L,lam,a,dt,totpts,vb):
	"""
	"""
	me = me0+".plot_traj: "

	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	## ------------------------------------------------------------------------
	## LINE COLOUR
	## Constant colour for each line segment of CH points
	## Colour decays away and hits min value.
	
	cm = plt.get_cmap("GnBu")
	
	CH = int(a/dt/5)	## Points in a colouring chunk
	NCH = totpts/CH ## Total number of chunks
	NCHC = 50 ## Number of coloured chunks, after which line is cmin
	cmin = 0.3
	
	## colourlist is linearly declining until NCHC chunks have been coloured, then constant colour
	colourlist = [cm(max(1.*(NCHC-i)/(NCHC-1),cmin)) for i in range(NCHC)] + [cm(cmin)]*(NCH-NCHC)
	
	ax.set_prop_cycle("color", colourlist) #if system()=="Linux" else ax.set_color_cycle(colourlist)
	## ------------------------------------------------------------------------	

	x = x[::-1]
	ex = ex[::-1]
	for i in range(totpts/CH):
		xseg = x[i*CH:(i+1)*CH+1]
		exseg = ex[i*CH:(i+1)*CH+1]
		ax.plot(xseg,exseg, "x")
							
	## Plot walls
	xx = np.linspace(0,lam,1000)
	U = -sp.integrate.cumtrapz(force_rect(xx,L,lam), xx, initial=0.0)
	ax.plot(xx, U, "k--", label=r"$U(x)$")
	
	ax.set_xlim(0,lam)
	ax.set_ylim(-3/a**0.5,+3/a**0.5)
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	ax.grid()
	
	return fig
	
##=============================================================================

def boundary_sim(xini, exini, a, x_step, dt, tmax, expmt):

	## Initialisation
	nstp = int(tmax/dt)
	
	## Simulate eta
	ex = sim_eta(exini, expmt, nstp, a, dt)
	x = np.zeros(nstp); x[0] = xini
	
	## Calculate trajectory
	for i in xrange(0,nstp-1):
		x[i+1] = x[i] + x_step(x[i],ex[i])
			
	return x, ex
		
## ====================================================================
def eul(x, ex, fx, dt):
	"""
	Euler step.
	Basic routine with all dependencies.
	"""
	return dt * ( fx(x) + ex )

## ====================================================================
def force_rect(x,L,lam):
	"""
	"""
	x = x%lam
	fx = -1.0*x*(x<=L)+(L/(lam-L))**2*(lam-x)*(x>L)
	return fx
	
## ====================================================================
## ====================================================================
if __name__=="__main__":
	main()