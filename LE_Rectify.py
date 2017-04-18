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
	
	Call with -a option to simulate and plot a PDF.
	Specify L (length of one limb) and lam (period).
	"""
	me = me0+".main: "
	t0 = time.time()

	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-a','--alpha',
		dest="alpha", type="float", default=-1.0)
	parser.add_option('-L',
		dest="L", type="float", default=0.8)
	parser.add_option('-l','--lam',
		dest="lam", type="float", default=-1.0)
	parser.add_option('-t','--tfac',
		dest="tfac", type="float", default=1.0)
	parser.add_option('--dt',
		dest="dt", type="float", default=0.01)
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
#	if lam>0 and L<lam: raise IOError, me+"Think about the geometry."
	tmax = 15e2*opt.tfac
	dt = opt.dt
	nosave = opt.nosave
	
	# outdir = "/home/users2/cs3006/Documents/Coloured_Noise/RECT/RECT_CAR_RL_t%.1f/"%(opt.tfac)
	outdir = "./Pressure/RECT/RECT_CAR_RL_t%.1f/"%(opt.tfac)
	if nosave:
		opt.showfig = True
	else:
		if not os.path.isdir(outdir):
			os.mkdir(outdir)
			print me+"Created directory",outdir
	
	## Make many wind plots
	if lam<0.0:
		if vb: print me+"Generating wind for many lambda with default parameter lists."
		for lam in [5.0]:
			plot_wind(lam,tmax,dt,outdir,nosave,vb)
	
	## Plot wind
	elif a<0.0:
		if vb: print me+"No alpha given. Generating wind diagram with default parameter lists."
		plot_wind(lam,tmax,dt,outdir,nosave,vb)
	
	## Plot density
	else:
		if vb: print me+"alpha given. Simulating and plotting density."
		plot_density(a, L, lam, outdir, tmax, dt, nosave, vb)

	if vb:	print me+"Total time %.1f seconds."%(time.time()-t0)

	if opt.showfig:	plt.show()
	
	return
	
##=================================================================================================
def sim(a,L,lam,tmax,dt,nosave,vb):
	"""
	Simulate OUP in asymmetric potential.
	"""
	me = me0+".sim: "
	t0 = time.time()
	
	if vb:	print me+"Simulating [a,L,lam]=[%.1g,%.1f,%.1f]"%(a,L,lam)

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
	
	if vb:	print me+"Winding rate %.3g"%(wind)
	if vb:	print me+"Simulation [a,L,lam]=[%.1f,%.1f,%.1f]: %.1f seconds."%(a,L,lam,time.time()-t0)
		
	return x, ex, wind
	
##=================================================================================================
def plot_density(a, L, lam, outdir, tmax, dt, nosave, vb, ax=None):
	"""
	Given trajectory, plot the density of points.
	"""
	me = me0+".plot_density: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh using phase_calc
	
	histfile = outdir+"/BHIS_CAR_RL_a%.1f_L%.1f_lam%.1f_dt%.1g.npz"%(a,L,lam,dt)
	
	try:
		data = np.load(histfile)
		print me+"Data file found:",histfile
		xbins, ebins, H, wind = data["xbins"], data["ebins"], data["H"], data["wind"]
		del data
		
	except IOError:
		print me+"No data found for alpha=%.1f, L=%.1f, lambda=%.1f. Calculating."%(a,L,lam)
		## Simulate data
		x, ex, wind = sim(a,L,lam,tmax,dt,nosave,vb)
		## Bins
		xbins = np.linspace(0.0,lam,200*lam+1)
		emax = 3/np.sqrt(a)
		ebins = np.linspace(-emax,+emax,100*emax+1)
		## Density
		H = np.histogramdd([x,ex],bins=[xbins,ebins],normed=True)[0]
		np.savez(histfile, xbins=xbins, ebins=ebins, H=H, wind=wind)
	
	## ------------------------------------------------------------------------
	
	## Force and potential
	cuspind = np.abs(xbins-L).argmin()
	fx = force_rect(xbins, L, lam)
	U  = -sp.integrate.cumtrapz(fx, xbins, initial=0.0); U-=U.min()
	
	## ----------------------------------------------------------------
	## Plotting
	
	if ax==None:
		print me+"No axes given; initialising new plot."
		fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	else:
		nosave = True
		
	plt.set_cmap("Greys")
	
	## Plot data
	xc = 0.5*(xbins[:-1]+xbins[1:])
	ec = 0.5*(ebins[:-1]+ebins[1:])
	
	## ------------------------------------------------------------------------
	
	## TWEAK COORDINATES
	## x is given such that 0 os at the cusp of the potential, and the minimum potential is on the edges
	## Roll coordinates to plot with potential minimum at "centre" and cusp on edge
	## Also reflect to make coordinates consistent with those in CN1 paper
	cuspind = np.abs(xbins-L).argmin()
	H = np.roll(H, fx.size-cuspind, axis=0)[::-1,::-1]
	fx = -np.roll(fx, fx.size-cuspind, axis=0)[::-1]
	## cuspind is now the position of the minimum
	
	## EXTEND to show more than one period
	repind = xc.size/4
	H = np.vstack([H[xc.size-repind:],H,H[:repind]])
	fx = np.hstack([fx[xc.size-repind:],fx,fx[:repind]])
	dx = xc[1]-xc[0]
	xbins = np.linspace(xbins[0]-repind*dx, xbins[-1]+repind*dx, xbins.size+2*repind+1)
	xc = np.linspace(xc[0]-repind*dx, xc[-1]+repind*dx, xc.size+2*repind)
	
	"""
	##----------------------
	## Plot x pdf
	Q = np.trapz(H, ec, axis=1)
	Q /= np.trapz(Q, xc)
	ax.plot(xc, Q-fx.max())
	# print np.trapz(Q[:cuspind], xc[:cuspind]), np.trapz(Q[cuspind:], xc[cuspind:])
	"""
	
	##----------------------
	
	## Plot x-eta pdf
	ax.contourf(xc,ec,H.T)

	## Force
	ax.plot(xbins, -fx, "k-", label=r"$-f(x)$")
	
	## Force ticks		
	ax.set_xticks([0,0.5*lam,L,lam])
	ax.set_xticklabels([r"$-L$",r"",r"$0$",r"$\ell$"], fontsize=fs["fsa"])
	ax.set_yticks([-L,0,+L**2/(lam-L)])
	ax.set_yticklabels([r"$-\frac{2U_0}{L}$",r"$0$",r"$\frac{2U_0}{\ell}$"], fontsize=fs["fsa"]+2)
	
	## CURRENT ARROWS
	## Using streamplot
	if 0:
		## Space grid
		X, ETAX = np.meshgrid(xc,ec, indexing="ij")
		F = 0.5*(fx[:-2]+fx[2:]).repeat(ec.size).reshape([xc.size,ec.size])
		## Currents
		Jx = (F + ETAX)*H
		Je = 1/a*ETAX*H + 1/(a*a)*np.gradient(H, ec[1]-ec[0])[1]
		with np.errstate(divide='ignore', invalid='ignore'): Vx, Ve = Jx/H, Je/H
		## Smooth
		Vx = np.nan_to_num(sp.ndimage.gaussian_filter(Vx, 5.0, order=0))
		Ve = np.nan_to_num(sp.ndimage.gaussian_filter(Ve, 5.0, order=0))
		## Plot arrows
		# sx, se = 70, 20
		# ax.quiver(xc[::sx], ec[::se], Vx.T[::se,::sx], Ve.T[::se,::sx] , scale=20/a, units='x', width=0.011)
		ax.streamplot(xc, ec, Vx.T, Ve.T,
						density=1.0, minlength=1.2, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
#		## Split into sections -- un-rolled
#		ax.streamplot(xc[:cuspind], ec, Vx[:cuspind].T, Ve[:cuspind].T,
#						density=0.8, minlength=1.8, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
#		ax.streamplot(xc[cuspind:], ec, Vx[cuspind:].T, Ve[cuspind:].T,
#						density=0.4, minlength=1, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
		## Add high-eta arrows
		if 0:
			fpind, fmind = np.abs(ec+fx.min()).argmin(), np.abs(ec-fx.max()).argmin()
			upind, dnind = fpind+80, fmind
			upwid, dnwid = ec.size-upind, dnind
			Vxup = Vx.max()*np.linspace(0,1,upwid)*np.ones([xc.size,upwid])
			Veup = np.outer(1.0*fx[1:], np.random.normal(0.0,4.0,upwid) )			ax.streamplot(xc, ec[upind:], Vxup.T, Veup.T,
							density=0.1, minlength=1, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
		#	Vxdn = Vx.min()*np.linspace(1,2,dnwid)*np.ones([xc.size,dnwid])
		#	Vedn = np.outer(-1.0*fx[:-1], np.random.normal(0.0,1.0,dnwid) )		#	ax.streamplot(xc, ec[:dnind], Vxdn.T, Vedn.T,
		#					density=0.5, minlength=2.5, color="orange", arrowsize=4, arrowstyle="-|>", zorder=2)
		## Draw them on
		else:
			ax.plot(np.hstack([xc[::10],xc[-1]]), fx.max()*1.01+np.random.normal(0.0,0.0015*fx.max(), xc.size/10+1), c="orange")
			ax.arrow(xc[xc.size/2], fx.max()*1.01, 0.1, 0.0,
					head_width=2*0.19, head_length=2*0.06, length_includes_head=True, overhang=0.3, fc='orange', ec='orange')
			ax.plot(xc, -fx.max()*1.08+np.random.normal(0.0,0.0015*fx.max(), xc.size), c="orange")
			ax.arrow(xc[xc.size/2], -fx.max()*1.08, -0.1, 0.0,
					head_width=2*0.19, head_length=2*0.06, length_includes_head=True, overhang=0.3, fc='orange', ec='orange')
	## Plot manually
	elif 1:
		## Space grid
		X, ETAX = np.meshgrid(xc,ec, indexing="ij")
		F = 0.5*(fx[:-2]+fx[2:]).repeat(ec.size).reshape([xc.size,ec.size])
		## Currents
		Jx = (F + ETAX)*H
		Je = -1/a*ETAX*H - 1/(a*a)*np.gradient(H, ec[1]-ec[0])[1]
		with np.errstate(divide='ignore', invalid='ignore'): Vx, Ve = Jx/H, Je/H
		## Smooth
		Vx = np.nan_to_num(sp.ndimage.gaussian_filter(Vx, 5.0, order=0))
		Ve = np.nan_to_num(sp.ndimage.gaussian_filter(Ve, 5.0, order=0))
		## Plot arrows
		ax.streamplot(xc, ec, Vx.T, Ve.T,
						density=1, minlength=1., color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
						# start_points = [[-0.2,0.0],[-0.5,0.0],[-1.0,0.0]])
#		## Split into sections -- un-rolled
#		ax.streamplot(xc[:cuspind], ec, Vx[:cuspind].T, Ve[:cuspind].T,
#						density=0.8, minlength=1.8, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
#		ax.streamplot(xc[cuspind:], ec, Vx[cuspind:].T, Ve[cuspind:].T,
#						density=0.4, minlength=1, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)

	## ------------------------------------------------------------------------

	## Figure
	ax.set_ylim(-L**2/(lam-L)*1.1,+L**2/(lam-L)*1.1)
	ax.set_axis_bgcolor(plt.get_cmap()(0.05))
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	ax.xaxis.labelpad = -15
	ax.yaxis.labelpad = -30
	ax.grid()
			
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = outdir+"/PDFxex_a%.1f_L%.1f_l%.1f."%(a,L,lam)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Density plot saved to",plotfile
	
	if vb:	print me+"Plotting and saving %.1f seconds."%(time.time()-t0)
		
	return

##=================================================================================================
def plot_wind(lam,tmax,dt,outdir,nosave,vb):
	"""
	Run several simulations to see how winding number changes.
	"""
	me = me0+".plot_wind: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh using phase_calc
	
	windfile = outdir+"/WIND_lam%.1f.npz"%(lam)
	
	try:
		data = np.load(windfile)
		print me+"Data file found:",windfile
		A, L, W = data["A"], data["L"], data["W"]
		del data
		
	except IOError:
		print me+"No data found for lambda=%.1f. Calculating."%(lam)
	
#		alist = [0.05,0.02,0.1]
		alist = [0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
#		alist = [0.2,0.5,1.0,1.5,2.0,3.0,4.0,5.0,10.0]
		Llist = lam*np.linspace(0.5,0.9,5)
		W = np.zeros([len(alist),len(Llist)])
		
		A, L = [], []
		for i,ai in enumerate(alist):
			for j,Li in enumerate(Llist):
				A += [ai]
				L += [Li]
				x, ex, W[i,j] = sim(ai,Li,lam,tmax,dt,vb)
		
		W = np.array(W)
		np.savez(windfile, A=A, L=L, W=W)
	
	## ------------------------------------------------------------------------
	## Fiddle
	
	if 0.0 not in A:
		numL = np.unique(L).size
		A = np.hstack([[0.0]*numL,A])
		L = np.hstack([L[:numL],L])
		W = np.vstack([[0.0]*numL,W])
		
	## Change definition of alpha
	## alpha_original = k\tau/\zeta = 2U_0\tau/L\zeta
	## alpha_new = 2U_0\tau/(L+\ell)\zeta
	A *= (lam)/L
	
	## Reorganise?
		
	W = W.flatten()
	
	W[L/lam==0.5]*=0.1
	
	## ------------------------------------------------------------------------
	## Plotting
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	afine = np.linspace(0.001,A.max()+0.001,1001)	## New alpha unit
	
	## Plot several lines
	for Li in np.unique(L):
		idx = (L==Li)
		ax.plot(A[idx], W[idx], "o-", label=r"$%.1f$"%((lam-Li)/lam))
		## Prediction. Convert back to old alpha unit.
		ax.plot(afine, calc_current(afine*Li/lam,Li,lam), c=ax.lines[-1].get_color(), ls="--")
		
	leg = ax.legend(loc=(0.51,0.17),ncol=2, fontsize=fs["fsl"]-1)
	leg.set_title(r"$\ell/(L+\ell)$", prop={"size":fs["fsl"]})
	leg.get_frame().set_alpha(0.7)
	ax.set_xlim(0.0,1.2)
	ax.set_ylim(bottom=-0.1e-3)
		
	## Figure
#	ax.xaxis.set_major_locator()
	ax.set_xlabel(r"$\frac{2U_0}{\zeta(L+\ell)^2}\tau$")
	ax.set_ylabel(r"$J / \frac{2U_0}{\zeta(L+\ell)^2}$")
#	ax.set_title(r"$\lambda=%.1f$"%(lam), fontsize=fs["fsa"])
	ax.grid()
	
#	plt.subplots_adjust(left=0.13)
	
	##-------------------------------------------
	## Inset density and streamlines
	if 1:
		left, bottom, width, height = [0.42,0.53,0.46,0.35]
		axin = fig.add_axes([left, bottom, width, height])
		plot_density(0.5, 1.4, 2.0, outdir, 300, 0.01, True, vb, axin)
		
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = outdir+"/WIND_lam%.1f."%(lam)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Density plot saved to",plotfile
	
	if vb:	print me+"Plotting and saving %.1f seconds."%(time.time()-t0)
		
	return

	
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
def calc_current(a,L,lam):
	"""
	Use stitched E2 to approximate density and evaluate current at cusp.
	See WM notebook.
	"""
	b = a*(L/(lam-L))**2
	norm = np.pi/(a)**0.5 * ( sp.special.erf((0.5*(1+a))**0.5 * L)/(1+a) + \
				+ ((lam-L)/L)*sp.special.erf((0.5*(1+b))**0.5 * L)/(1+b) )
	Jp = np.exp(-0.5*(1+a)*L**2)/(a*(1+a)) / norm
	Jm = np.exp(-0.5*(1+b)*L**2)/(a*(1+b)) / norm
	return Jp-Jm

## ====================================================================
## ====================================================================
if __name__=="__main__":
	main()
