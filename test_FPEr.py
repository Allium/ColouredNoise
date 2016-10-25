me0 = "test_FPEr"

import numpy as np
import scipy.interpolate, scipy.ndimage, scipy.optimize
from sys import argv
import os, optparse, glob, time
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable as mplmal

from LE_SBS import force_const, force_lin, force_dcon, force_dlin,\
					force_tan, force_dtan, force_nu, force_dnu
from LE_Utils import filename_par

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

"""
Does the derived density in r and etar satisfy the radial FPE derived in 23/09/2016
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
	me = me0+"plot_FPEres: "

	## Get pars from filename
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
				
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins  = bins["rbins"]
	erbins = bins["erbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	etar = 0.5*(erbins[1:]+erbins[:-1])

	## Load histogram
	H = np.load(histfile)
	try:				H = H.sum(axis=2)	## If old _phi file
	except ValueError: 	pass
	
	## ------------------------------------------------------------------------
	
	## Normalise and convert to density
	H /= np.trapz(np.trapz(H,etar,axis=1),r,axis=0)
	rho = H / ( (2*np.pi)**2.0 * reduce(np.multiply, np.ix_(r,etar)) )
	
	## ------------------------------------------------------------------------

	## Force
	assert histfile.find("_DL_")
	ftype = "dlin"
	f = force_dlin(r,r,R,S)
		
	##
	rr = r[:,np.newaxis]
	ee = etar[np.newaxis,:]
	ff = f[:,np.newaxis]
		
	## FPE operator ARRAY
	D = lambda arr, x, **kwargs: scipy.ndimage.gaussian_filter1d(arr, 1.0, axis=kwargs["axis"], order=1) / (x[1]-x[0])
	Drrho = scipy.ndimage.gaussian_filter1d(rho, 1.0, axis=0, order=1) / (r[1]-r[0])
	Derho = scipy.ndimage.gaussian_filter1d(rho, 1.0, axis=1, order=1) / (etar[1]-etar[0])
	DDerho = scipy.ndimage.gaussian_filter1d(rho, 1.0, axis=1, order=2) / (etar[1]-etar[0])**2
	res = -D((ee+ff)*rho, r, axis=0) -1/rr*(ee+ff)*rho + 1/a*D(ee*rho, etar, axis=1) + 1/a*rho +\
			+ 1/a**2*1/ee*Derho + 1/a**2*DDerho#D(D(rho,etar,axis=1),etar,axis=1)

	## ------------------------------------------------------------------------

	## Plotting
	fig, axs = plt.subplots(2,1, sharex=True)

	X, Y = np.meshgrid(r, etar)
	
	## Plot density
	ax = axs[0]
	Z = rho.T #/ rho.max()
	im = ax.contourf(X, Y, Z, 10, vmin=0.0, antialiased=True)
	cax = mplmal(ax).append_axes("right", size="5%", pad=0.05)
	fig.colorbar(im, cax=cax)\
	
	axs[0].set_ylabel(r"$\eta_r$", 	fontsize=18)
	ax.set_title("2D Radial Density", fontsize=14)
	
	## Plot residue
	ax = axs[1]
	Z = res.T #/ np.abs(res).max()
	clim = float("%.1g"%(Z.mean()+2*Z.std()))
	im = ax.contourf(X, Y, Z, levels=np.linspace(-clim,clim,11), cmap=cm.BrBG, antialiased=True)
	cax = mplmal(ax).append_axes("right", size="5%", pad=0.05)
	fig.colorbar(im, cax=cax)
	
	axs[1].set_xlabel(r"$r$", 		fontsize=18)
	axs[1].set_ylabel(r"$\eta_r$", 	fontsize=18)
	ax.set_title("Radial FP Operator On Density", fontsize=14)
	
	## Indicate wall
	for ax in axs:
		ax.axvline(S, c="k", lw=2)
		ax.axvline(R, c="k", lw=2)

	if not nosave:
		plotfile = os.path.dirname(histfile)+"/FPEres"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+": Figure saved to",plotfile

	return
	
##=============================================================================
if __name__ == "__main__":
	input()
