me0 = "LE_C2PDF"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt

from LE_CSim import force_dlin, force_clin, force_mlin, force_nlin
from LE_Utils import filename_par, fs, set_mplrc

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## Plot defaults
set_mplrc(fs)

## ============================================================================

def main():
	"""
	Adapted from LE_CPDF.py.
	"""
	me = me0+".main: "
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
	vb = opt.verbose
	
	assert "_CAR_" in args[0], me+"Functional only for Cartesian geometry."
	assert "_UL_" in args[0], me+"Functional only for ulin ftype."
	
	## Plot file
	if os.path.isfile(args[0]):
		plot_pdf2d(args[0], nosave, vb)
	## Plot all files
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_CAR_U*"+searchstr+"*.npy"))
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pdf2d(histfile, nosave, vb)
			plt.close()
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

	
##=============================================================================
def plot_pdf2d(histfile, nosave, vb):
	"""
	Read in data for a single file and plot 2D PDF projections.
	"""
	me = me0+".plot_pdf2D: "
	t0 = time.time()

	##-------------------------------------------------------------------------
	
	## Get pars from filename
	
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T")
	
	##-------------------------------------------------------------------------
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	ybins = bins["ybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	y = 0.5*(ybins[1:]+ybins[:-1])
	
	
	## Wall indices
	Rind = np.abs(x-R).argmin()
	
	##-------------------------------------------------------------------------
	
	## Histogram / density
	H = np.load(histfile)
	rho = H / (H.sum() * (x[1]-x[0])*(y[1]-y[0]))
	
	## ------------------------------------------------------------------------
	
	## Plotting
	
	fig, axs = plt.subplots(1,2, sharey=True, figsize=fs["figsize"])
	fig.canvas.set_window_title("2D PDF")
	lvls = 15
	
	plt.rcParams["image.cmap"] = "coolwarm"#"Greys"#
	
	## ------------------------------------------------------------------------
	
	## Plot density
	ax = axs[0]
	ax.contourf(x, y, rho.T, lvls)
	
	## Indicate bulk
	yfine = np.linspace(y[0],y[-1],1000)
	ax.scatter(+R+S*np.sin(2*np.pi*yfine/T), yfine, c="k", s=1)
	
	ax.set_xlim(xbins[0],xbins[-1])
	ax.set_ylim(ybins[0],ybins[-1])
	
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	ax.set_title(r"$\rho(x,y)$ data", fontsize=fs["fsa"])
	
	## ------------------------------------------------------------------------
	
	## Plot WN density
	ax = axs[1]
	
	X, Y = np.meshgrid(x, y, indexing="ij")
	U = 0.5*(X-R-S*np.sin(2*np.pi*Y/T))**2 * (X>R+S*np.sin(2*np.pi*Y/T))
	rho_WN = np.exp(-U) / np.trapz(np.trapz(np.exp(-U), y, axis=1), x, axis=0)
	ax.contourf(x, y, rho_WN.T, lvls)
	
	## Indicate bulk
	yfine = np.linspace(y[0],y[-1],1000)
	ax.scatter(+R+S*np.sin(2*np.pi*yfine/T), yfine, c="k", s=1)
	
	ax.set_xlim(xbins[0],xbins[-1])
	ax.set_ylim(ybins[0],ybins[-1])
	
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	# ax.set_ylabel(r"$y", fontsize=fs["fsa"])
	ax.set_title(r"$\rho(x,y)$ WN", fontsize=fs["fsa"])
		
	## ------------------------------------------------------------------------
	
	title = r"PDF projections. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f$"%(a,R,S,T)
	fig.suptitle(title, fontsize=fs["fst"])
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFxy2d"+os.path.basename(histfile)[4:-4]
		plotfile += "."+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)

	return
	
	

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
