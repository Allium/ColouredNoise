me0 = "LE_Rectify"

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
	parser.add_option("-a","--alpha",
		dest="alpha", type="float", default=-1.0)
	parser.add_option("-u",
		dest="u", type="float", default=-1.0)
	parser.add_option("-l","--lam",
		dest="lam", type="float", default=0.75)
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
	u = opt.u
	lam = opt.lam
	tmax = 15e2*opt.tfac
	dt = opt.dt
	nosave = opt.nosave
	
	assert 0<lam<1.0, me+"lam parameter 0<lam<0.5."
	if lam>0.5: lam = 1.0-lam
	
	# outdir = "/home/users2/cs3006/Documents/Coloured_Noise/RECT/RECT_CAR_RL_t%.1f/"%(opt.tfac)
	outdir = "./Pressure/RECT/RECT_CAR_RL_t%.1f/"%(opt.tfac)
	if nosave:
		opt.showfig = True
	else:
		if not os.path.isdir(outdir):
			os.mkdir(outdir)
			print me+"Created directory",outdir
	
	## Make many wind plots
	if u<0.0:
		if vb: print me+"Generating wind for many u with default parameter lists."
		for u in [0.5,1.0,2.0]:
			plot_wind(u,tmax,dt,outdir,nosave,vb)
	
	## Plot wind
	elif a<0.0:
		if vb: print me+"No alpha given. Generating wind diagram with default parameter lists."
		plot_wind(u,tmax,dt,outdir,nosave,vb)
	
	## Plot density
	else:
		if vb: print me+"alpha given. Simulating and plotting density."
		plot_density(a, u, lam, outdir, tmax, dt, nosave, vb)

	if vb:	print me+"Total time %.1f seconds."%(time.time()-t0)

	if opt.showfig:	plt.show()
	
	return
	
	
##=================================================================================================
def plot_density(a, u, lam, outdir, tmax, dt, nosave, vb, ax=None):
	"""
	Given trajectory, plot the density of points.
	"""
	me = me0+".plot_density: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh using phase_calc
	
	histfile = outdir+"/BHIS_CAR_RL_a%.1f_u%.1f_lam%.1f_dt%.1g.npz"%(a,u,lam,dt)
	
	try:
		data = np.load(histfile)
		print me+"Data file found:",histfile
		xbins, ebins, H, wind = data["xbins"], data["ebins"], data["H"], data["wind"]
		del data
		
	except IOError:
		print me+"No data found for alpha=%.1f, u=%.1f, lambda=%.1f. Calculating."%(a,u,lam)
		## Simulate data
		x, ex, wind = sim(a,u,lam,tmax,dt,vb)
		## Bins
		xbins = np.linspace(0.0,u,int(400*u+1))
		emax = 3/(u*np.sqrt(a))
		ebins = np.linspace(-emax,+emax,int(100*emax+1))
		## Density
		H = np.histogramdd([x,ex],bins=[xbins,ebins],normed=True)[0]
		np.savez(histfile, xbins=xbins, ebins=ebins, H=H, wind=wind)
	
	## ------------------------------------------------------------------------
	
	## Force and potential
	cuspx = u*lam
	cuspind = np.abs(xbins-cuspx).argmin()
	fx = force_rect(xbins, u, lam)
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
	cuspind = np.abs(xbins-cuspx).argmin()
	H = np.roll(H, fx.size-cuspind, axis=0)
	fx = np.roll(fx, fx.size-cuspind, axis=0)
	## cuspind is now the position of the minimum
#	
#	## EXTEND to show more than one period
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
	ax.set_xticks([0, 0.5*u, (1-lam)*u, u])
	ax.set_xticklabels([r"$-L$", r"", r"$0$", r"$+\ell$"], fontsize=fs["fsa"])
	ax.set_yticks([-2/(1-lam), 0, 2/(lam)])
	ax.set_yticklabels([r"$-\frac{2U_0}{L}$", r"$0$", r"$\frac{2U_0}{\ell}$"], fontsize=fs["fsa"]+2)
	
	##----------------------
	
	## CURRENT ARROWS
	## Using streamplot
	if 0:
		## Space grid
		X, ETAX = np.meshgrid(xc,ec, indexing="ij")
		F = 0.5*(fx[:-2]+fx[2:]).repeat(ec.size).reshape([xc.size,ec.size])
		## Currents
		Jx = (F + ETAX)*H
		Je = + 1/a*ETAX*H + 1/(a*a)*np.gradient(H, ec[1]-ec[0])[1]
		with np.errstate(divide='ignore', invalid='ignore'): Vx, Ve = Jx/H, Je/H
		## Smooth
		Vx = np.nan_to_num(sp.ndimage.gaussian_filter(Vx, 5.0, order=0))
		Ve = np.nan_to_num(sp.ndimage.gaussian_filter(Ve, 5.0, order=0))
		## Plot arrows
		# sx, se = 70, 20
		# ax.quiver(xc[::sx], ec[::se], Vx.T[::se,::sx], Ve.T[::se,::sx] , scale=20/a, units='x', width=0.011)
		ax.streamplot(xc, ec, Vx.T, Ve.T,
						density=1.0, minlength=0.5, color="orange", arrowsize=4, arrowstyle="fancy", zorder=2)
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
		elif 0:
			ax.plot(np.hstack([xc[::10],xc[-1]]), fx.max()*1.01+np.random.normal(0.0,0.0015*fx.max(), xc.size/10+1), c="orange")
			ax.arrow(xc[xc.size/2], fx.max()*1.01, 0.1, 0.0,
					head_width=2*0.19, head_length=2*0.06, length_includes_head=True, overhang=0.3, fc='orange', ec='orange')
			ax.plot(xc, -fx.max()*1.08+np.random.normal(0.0,0.0015*fx.max(), xc.size), c="orange")
			ax.arrow(xc[xc.size/2], -fx.max()*1.08, -0.1, 0.0,
					head_width=2*0.19, head_length=2*0.06, length_includes_head=True, overhang=0.3, fc='orange', ec='orange')
	## Plot Ev2 ellipses
	## PDF points in "wrong" direction
	elif 0:
		## Offset x coordinate here for clarity
		xcurr = np.linspace(xc[0],xc[-1],xc.size*10+1)-u*(1-lam)
		c = [0.5,1.0,2.0]
		for ci in c:
			zind = np.abs(xcurr).argmin()
			## Plotting ellipses. See 19/04/2017.
			## Right hand side, l
			discr = -xcurr[zind:]*xcurr[zind:]/(a) + ci*u*lam*lam/(a/(lam*lam)*(a/(lam*lam)+1))
			eli = 1/(u*lam*lam) * np.hstack([xcurr[zind:]+np.sqrt(discr), xcurr[zind:]-np.sqrt(discr)])
			ax.plot(np.hstack([xcurr[zind:],xcurr[zind:]])+u*(1-lam), eli, "-", color="orange")
			## Left hand side, L
			ci *= (a/(1-lam)**2 + 1)/(a/lam**2 + 1)
			discr = -xcurr[:zind]*xcurr[:zind]/(a) + ci*u*(1-lam)*(1-lam)/(a/((1-lam)*(1-lam))*(a/((1-lam)*(1-lam))+1))
			eli = 1/(u*(1-lam)*(1-lam)) * np.hstack([xcurr[:zind]+np.sqrt(discr), xcurr[:zind]-np.sqrt(discr)])
			ax.plot(np.hstack([xcurr[:zind],xcurr[:zind]])+u*(1-lam), eli, "-", color="orange")
	## Plot manually THIS IS AWFUL
	elif 1:
		## a0.5_u0.5_lam0.3
		ptslist = [
			## Outer loop
			[[0.09,-2.56],[0.1,-2.0],[0.15,-0.9],[0.2,-0.2],[0.243,0.352],[0.258,0.502],[0.3,1.1],[0.35,2.1],[0.4,3.8],[0.45,5.6],[0.47,6.1],[0.48,5.55]],
			[[0.48,5.551],[0.473,4.66],[0.45,3.14],[0.43,1.87],[0.40,0.71],[0.38,-0.38],[0.36,-1.44],[0.32,-2.2],[0.24,-2.58],[0.15,-3.1],[0.11,-3.02],[0.1,-2.9],[0.09,-2.56]],
			## Inner loop
			[[0.21,-1.5],[0.22,-1.0],[0.25,-0.5],[0.3,0.3],[0.35,1.0],[0.4,3.0],[0.45,4.7],[0.46,4.6]],
			[[0.21,-1.5],[0.22,-1.8],[0.25,-1.9],[0.3,-1.6],[0.35,-1.1],[0.4,1.4],[0.45,3.9],[0.46,4.6]],
			## Lower run
			# [[0.626,-4.18],[0.53,-4.23],[0.4,-3.5],[0.15,-3.8],[0.05,-4.02],[-0.02,-4.03],[-0.06,-3.95],[-0.1248,-3.78]],
			# [[0.626,-6.1],[0.53,-6.13],[0.4,-5.9],[0.15,-6.0],[0.05,-6.1],[-0.02,-6.1],[-0.06,-6.03],[-0.1248,-5.97]],
			[[1.1248,-3.78],[0.55,-4.02],[0.48,-4.03],[0.43,-3.95],[0.3752,-3.78],[0.15,-3.8],[0.05,-4.02],[-0.02,-4.03],[-0.06,-3.95],[-0.1248,-3.78]],
			[[0.65,-6.0],[0.55,-6.1],[0.48,-6.1],[0.44,-6.03],[-0.3752,-5.97],[0.4,-5.9],[0.15,-6.0],[0.05,-6.1],[-0.02,-6.1],[-0.06,-6.03],[-0.1248,-5.97]],
			## Upper run
			[[-0.125,6.5],[-0.05,6.8],[0.01,6.7],[0.1,5.5],[0.2,5.7],[0.3,6.0],[0.4,6.4],[0.5,6.85],[0.6,6.2],[0.626,6.0]],
			## Left loop
			# [[-0.124,3.56],[-0.09,4.2],[-0.06,4.9],[-0.044,4.75]],
			# [[-0.125,-0.211],[-0.1,0.91],[-0.09,1.4],[-0.058,2.88],[-0.048,4.0],[-0.046,4.3],[-0.044,4.74]],
			[[-0.2,1.1],[-0.15,2.1],[-0.1,3.8],[-0.05,5.6],[-0.03,6.1],[-0.02,5.55]],
			[[-0.02,5.551],[-0.027,4.66],[-0.05,3.14],[-0.07,1.87],[-0.10,0.71],[-0.12,-0.38],[-0.14,-1.44],[-0.18,-2.2]],
			[[-0.2,0.3],[-0.15,1.0],[-0.1,3.0],[-0.05,4.7],[-0.04,4.6]],
			[[-0.2,-1.6],[-0.15,-1.1],[-0.1,1.4],[-0.05,3.9],[-0.04,4.6]],
			## Right loop
			# [[0.567,-2.58],[0.573,-2.0],[0.59,-1.3],[0.62,-0.85],[0.624,-0.8]],
			# [[0.624,-3.2],[0.6,-3.2],[0.58,-3.0],[0.567,-2.58]]
			[[0.59,-2.56],[0.6,-2.0],[0.65,-0.9],[0.7,-0.2]],
			[[0.74,-2.58],[0.65,-3.1],[0.61,-3.02],[0.6,-2.9],[0.59,-2.56]]
			]
		for pts in ptslist:
			pts = np.array(pts).T
			ycurr = sp.interpolate.interp1d(pts[0],pts[1], kind="cubic", bounds_error=False, fill_value=np.nan)(xc)
			ax.plot(xc,ycurr, "-", color="orange", zorder=1)
			# ax.plot(pts[0],pts[1], "o", ms=4)
		## Outer loop
		# ax.arrow(0.243, 0.352, 0.01, 0.12,
			# head_length=0.3, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		# ax.arrow(0.4, 0.71, -0.01, -0.5,
			# head_length=0.3, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(0.09, -2.7, 0, 0.0001,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(0.48, 5.7, 0, -0.0001,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		## Inner loop
		ax.arrow(0.212, -1.7, 0, 0.01,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(0.46, 4.6, 0, -0.01,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		## Lower run
		ax.arrow(0.05, -4.04, -0.001, -0.0,
			head_length=0.03, head_width=0.4, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(0.265, -5.87, -0.0001, -0.000,
			head_length=0.03, head_width=0.4, overhang=0.2, fc='orange', ec='orange',lw=0)
		## Upper run
		ax.arrow(0.115, 5.4, 0.001, 0.0,
			head_length=0.03, head_width=0.4, overhang=0.2, fc='orange', ec='orange',lw=0, zorder=2)
		## Left loop
		# ax.arrow(-0.044,4.75, 0, -0.01,
			# head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(-0.02, 5.7, 0, -0.0001,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(-0.04, 4.6, 0, -0.01,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		## Right loop
		# ax.arrow(0.57,-2.65, 0, +0.01,
			# head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		ax.arrow(0.595, -2.7, 0, 0.0001,
			head_length=0.4, head_width=0.03, overhang=0.2, fc='orange', ec='orange',lw=0)
		
	## ------------------------------------------------------------------------

	## Figure
	ax.set_ylim(-2/lam*1.1,+2/lam*1.1)
	ax.set_axis_bgcolor(plt.get_cmap()(0.08))
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	ax.xaxis.labelpad = -15
	ax.yaxis.labelpad = -30
	ax.grid()
			
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = outdir+"/PDFxex_a%.1f_u%.1f_lam%.1f."%(a,u,lam)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Density plot saved to",plotfile
	
	if vb:	print me+"Plotting and saving %.1f seconds."%(time.time()-t0)
		
	return

##=================================================================================================
def plot_wind(u,tmax,dt,outdir,nosave,vb):
	"""
	Run several simulations to see how winding number changes.
	"""
	me = me0+".plot_wind: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh using phase_calc
	## L is array of lambda values
	
	windfile = outdir+"/WIND_u%.1f.npz"%(u)
	
	try:
		data = np.load(windfile)
		print me+"Data file found:",windfile
		A, L, W = data["A"], data["L"], data["W"]
		del data
		
	except IOError:
		print me+"No data found for u=%.1f. Calculating."%(u)
	
		alist = [0.02]
		# alist = [0.02,0.05,0.1,0.15,0.25,0.3,0.35,0.4,0.5]
#		alist = [0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
		Llist = np.linspace(0.1,0.5,5)
		W = np.zeros([len(alist),len(Llist)])
		
		A, L = [], []
		for i,ai in enumerate(alist):
			for j,Li in enumerate(Llist):
				A += [ai]
				L += [Li]
				dt = 0.005 if ai<=0.15 else 0.01
				x, ex, W[i,j] = sim(ai,u,Li,tmax,dt,vb)
		
		W = np.array(W)
		np.savez(windfile, A=A, L=L, W=W)
	
	## ------------------------------------------------------------------------
	## Fiddle
	
	if 0.0 not in A:
		numL = np.unique(L).size
		A = np.hstack([[0.0]*numL,A])
		L = np.hstack([L[:numL],L])
		W = np.vstack([[0.0]*numL,W])
				
	## Change definition of alpha?
	
	## Reorganise
	W = W.flatten()
		
	## ------------------------------------------------------------------------
	## Plotting
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	afine = np.linspace(0.001,A.max()+0.001,1001)	## New alpha unit
	
	## Plot several lines
	for Li in np.unique(L):
		idx = (L==Li)
		ax.plot(A[idx], W[idx], "o-", label=r"$%.1f$"%(Li))
		## Prediction.
		ax.plot(afine, calc_current(afine*2.0,u,Li), c=ax.lines[-1].get_color(), ls="--")
		
	leg = ax.legend(loc=(0.48,0.56),ncol=2, fontsize=fs["fsl"]-1)
	leg.set_title(r"$\ell/(L+\ell)$", prop={"size":fs["fsl"]})
	leg.get_frame().set_alpha(0.7)
	ax.set_xlim(0.0,1.0)
	ax.set_ylim(bottom=1.1*W.min(),top=-0.1*W.min())
		
	## Figure
	ax.set_xlabel(r"$\tau\cdot\frac{2U_0}{\zeta(L+\ell)^2}$")
	ax.set_ylabel(r"$J / \frac{2U_0}{\zeta(L+\ell)^2}$")
	ax.grid()
	
	plt.subplots_adjust(left=0.15)
	
	##-------------------------------------------
	## Inset density and streamlines
	if 1:
#		left, bottom, width, height = [0.42,0.53,0.46,0.35]
		left, bottom, width, height = [0.35,0.16,0.535,0.35]
		axin = fig.add_axes([left, bottom, width, height])
		plot_density(0.5, 0.5, 0.3, outdir, 300, 0.01, True, vb, axin)
		
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = outdir+"/WIND_u%.1f."%(u)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Density plot saved to",plotfile
	
	if vb:	print me+"Plotting and saving %.1f seconds."%(time.time()-t0)
		
	return

	
##=================================================================================================
##=================================================================================================
def sim(a,u,lam,tmax,dt,vb):
	"""
	Simulate OUP in asymmetric potential.	"""
	me = me0+".sim: "
	t0 = time.time()
	
	if vb:	print me+"Simulating [a,u,lam]=[%.1g,%.1f,%.1f]"%(a,u,lam)

	## Injection x coordinate
	xini = 0.0
	eIC = 0.0
	
	## Functions to pass
	f_step = lambda x: force_rect(x, u, lam)
	x_step = lambda x, ex: eul(x, ex, f_step, dt)
	
	## ----------------------------------------------------------------
	## Simulation

	## Precompute exp(-t)
	expmt = np.exp((np.arange(-10*a,dt,dt))/a)

	x, ex = boundary_sim(xini, eIC, a, u, x_step, dt, tmax, expmt)
		
	## Boundary conditions
	wind = int(x[-1]/u)/tmax
	x = x%u
	
	if vb:	print me+"Winding rate %.3g"%(wind)
	if vb:	print me+"Simulation [a,u,lam]=[%.1f,%.1f,%.1f]: %.1f seconds."%(a,u,lam,time.time()-t0)
		
	return x, ex, wind
	
##=============================================================================

def boundary_sim(xini, exini, a, u, x_step, dt, tmax, expmt):

	## Initialisation
	nstp = int(tmax/dt)
	
	## Simulate eta -- note u
	ex = (1/u)*sim_eta(exini, expmt, nstp, a, dt)
	x = np.zeros(nstp); x[0] = xini
	
	## Calculate trajectory -- note u
	for i in xrange(0,nstp-1):
		x[i+1] = x[i] + u*x_step(x[i],ex[i])
			
	return x, ex
		
## ====================================================================
def eul(x, ex, fx, dt):
	"""
	Euler step.
	Basic routine with all dependencies.
	"""
	return dt * ( fx(x) + ex )

## ====================================================================
def force_rect(x,u,lam):
	"""
	See notes 18/04/2017
	"""
	x = x%u
	fx = -2/u*( 1/lam**(2)*x*(x<=lam*u) + 1/(1-lam)**(2)*(x-u) * (x>lam*u))
	return fx
	
## ====================================================================
def calc_current(a,u,lam):
	"""
	Use stitched E2 to approximate density and evaluate current at cusp.
	See 18/04/2017
	"""
	al = np.sqrt(a/lam**2 + 1)
	aL = np.sqrt(a/(1-lam)**2 + 1)
	norm = np.pi*a * ( sp.special.erf(u*al)/(np.sqrt(a)/lam * al) + \
				+ sp.special.erf(u*aL)/(np.sqrt(a)/(1-lam) * aL) )
	Jp = np.exp(-u*u*al*al)/(al) / norm
	Jm = np.exp(-u*u*aL*aL)/(aL) / norm
	return Jp-Jm

## ====================================================================
## ====================================================================
if __name__=="__main__":
	main()
