me0 = "test_FPEr"

import numpy as np
import scipy.interpolate, scipy.ndimage, scipy.optimize
from sys import argv
import os, optparse, glob, time
from matplotlib import cm

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mplmal

from LE_SBS import force_const, force_lin, force_dcon, force_dlin,\
					force_tan, force_dtan, force_nu, force_dnu
from LE_Utils import filename_par

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

fsa, fsl, fst = 18, 12, 14

"""
Does the derived density in r and etar satisfy the radial FPE.
Incorrect derivation 23/09/2016; more correct 25/10/2016.

py test_FPEr.py filename opts
"""

##=============================================================================
def input():
	"""
	Read command-line arguments and decide which plot to make.
	"""

	me = me0+".input: "
	t0 = time.time()
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('--str',
		dest="searchstr", default="", type="str")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	searchstr = opt.searchstr
	nosave = opt.nosave
	verbose = opt.verbose
		
	if os.path.isfile(args[0]):
		plot_FPEres(args[0], nosave, verbose)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		if verbose: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_FPEres(histfile, nosave, verbose)
			plt.close()
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_FPEres(histfile, nosave, vb):
	"""
	Read in data for a single file density, run it through FP operator, and plot.
	"""
	me = me0+".plot_FPEres: "

	## Get pars from filename
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	
	phifile = bool(histfile.find("_phi") + 1)
	psifile = bool(histfile.find("_psi") + 1)
				
	## ------------------------------------------------------------------------
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins  = bins["rbins"]
	erbins = bins["erbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	etar = 0.5*(erbins[1:]+erbins[:-1])
	if psifile:
		epbins = bins["epbins"]
		etap = 0.5*(epbins[1:]+epbins[:-1])

	## Force
	assert histfile.find("_DL_")
	ftype = "dlin"
	f = force_dlin(r,r,R,S)			
				
	## Spatial arrays with dimensions commensurate to rho
	if psifile:
		ff = f[:,np.newaxis,np.newaxis]
		rr = r[:,np.newaxis,np.newaxis]
		ee = etar[np.newaxis,:,np.newaxis]
		pp = etap[np.newaxis,np.newaxis,:]
		dV = (r[1]-r[0])*(etar[1]-etar[0])*(etap[1]-etap[0])	## Assumes regular grid
	else:
		ff = f[:,np.newaxis]
		rr = r[:,np.newaxis]
		ee = etar[np.newaxis,:]
		dV = (r[1]-r[0])*(etar[1]-etar[0])	## Assumes regular grid
	
	## ------------------------------------------------------------------------

	## Load histogram
	H = np.load(histfile)
	if phifile:
		H = H.sum(axis=2)	## If old _phi file
	
	## Normalise and convert to density
	H /= H.sum()*dV
	rho = H / ( (2*np.pi)**2.0 * rr*ee )
		
	## ------------------------------------------------------------------------
		
	## Derivative function
	D = lambda arr, x, order, **kwargs: \
			scipy.ndimage.gaussian_filter1d(arr, 1.0, order=order, axis=kwargs["axis"]) / (x[1]-x[0])
	
	## FPE operator
	t0 = time.time()
	if psifile:
		res = -D((ee*np.cos(pp)+ff)*rho, r, 1, axis=0) -1/rr*ff*rho + 1/rr*ee*np.sin(pp)*D(rho, etap, 1, axis=2) +\
				+ 1/a*D(ee*rho, etar, 1, axis=1) + 1/a*rho +\
				+ 1/a**2*1/ee*D(rho, etar, 1, axis=1) + 1/a**2*D(rho, etar, 2, axis=1) + 1/a**2*1/ee**2*D(rho, etap, 2, axis=2)
	else:
		## If radial (r,eta) only, or (r,eta,eta_phi) converted to (r,eta)
		res = -D((ee+ff)*rho, r, 1, axis=0) -1/rr*(ee+ff)*rho + 1/a*D(ee*rho, etar, 1, axis=1) +\
				+ 1/a*rho + 1/a**2*1/ee*D(rho, etar, 1, axis=1) +\
				+ 1/a**2*D(rho, etar, 2, axis=1)
	if vb: print me+"Residue calculation %.2g seconds."%(time.time()-t0)
			
	## ------------------------------------------------------------------------

	## Plotting
	fig, axs = plt.subplots(2,1, sharex=True)

	X, Y = np.meshgrid(r, etar)
	
	## ------------------------------------------------------------------------

	## Plot density
	ax = axs[0]
	if psifile:
		Z = np.trapz(rho, etap, axis=2).T
	else:
		Z = rho.T
	
	clim = float("%.1g"%(Z.max()))
	im = ax.contourf(X, Y, Z, levels=np.linspace(0,clim,11), vmin=0.0, antialiased=True)
	cax = mplmal(ax).append_axes("right", size="5%", pad=0.05)
	fig.colorbar(im, cax=cax)
	
	axs[0].set_ylabel(r"$\eta_r$", 	fontsize=fsa)
	ax.set_title(r"2D Radial Density. $\alpha=%.1f$, $R=%.1f$, $S=%.1f$."%(a,R,S), fontsize=fst)
	
	## ------------------------------------------------------------------------

	## Plot residue
	ax = axs[1]
	if psifile:
		Z = np.trapz(res, etap, axis=2).T
	else:
		Z = res.T
	clim = float("%.1g"%(Z.mean()+1.0*Z.std()))	## std dominated by choppy r=0.
	im = ax.contourf(X, Y, Z, levels=np.linspace(-clim,clim,11), cmap=cm.BrBG, antialiased=True)
	cax = mplmal(ax).append_axes("right", size="5%", pad=0.05)
	fig.colorbar(im, cax=cax)
	
	axs[1].set_xlabel(r"$r$", 		fontsize=fsa)
	axs[1].set_ylabel(r"$\eta_r$", 	fontsize=fsa)
	ax.set_title("Radially Symmetric FP Residual", fontsize=fst)
	
	## ------------------------------------------------------------------------
	
	## Indicate wall
	for ax in axs:
		ax.axvline(S, c="k", lw=2)
		ax.axvline(R, c="k", lw=2)

	## ------------------------------------------------------------------------

	if not nosave:
		plotfile = os.path.dirname(histfile)+"/FPEres"+os.path.basename(histfile)[4:-4]+"_psi"*psifile+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile

	return
	
##=============================================================================
if __name__ == "__main__":
	input()
