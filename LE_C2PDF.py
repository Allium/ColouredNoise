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
from matplotlib.ticker import MaxNLocator

from LE_CSim import force_dlin, force_clin, force_mlin, force_nlin
from LE_Utils import filename_par, fs, set_mplrc
from schem_force import plot_U3D_ulin, plot_U2D_ulin

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## Plot defaults
set_mplrc(fs)

## ============================================================================

def main():
	"""
	Adapted from LE_CPDF.py.
	
	OPTIONS / FLAGS
		slices	False	Plot PDF as a function of x for several y
		intx	False	Plot PDF integrated over x in wall region as a function of y
		showfig	False	Show the figure in matplotlib window
		plotall	False	Read in all files in directory (matching srchstr) and make plot
		srchstr	""		Only plot files matching string (for plotall)
		nosave	False	Do not save figure
		verbose	False	Status to stdout
	"""
	me = me0+".main: "
	t0 = time.time()
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-1','--1dplot','--slices','--multi',
		dest="slices", default=False, action="store_true")
	parser.add_option('--intx',
		dest="intx", default=False, action="store_true")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('--str',
		dest="srchstr", default="", type="str")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	slices = opt.slices
	intx = opt.intx
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
	nosave = opt.nosave
	vb = opt.verbose
	
	assert "_CAR_" in args[0], me+"Functional only for Cartesian geometry."
	assert "_UL_" in args[0], me+"Functional only for ulin ftype."
	
	## Plot file
	if os.path.isfile(args[0]):
		if slices:
			plot_pdf1d(args[0], nosave, vb)
		elif intx:
			plot_pdf1d_intx(args[0], nosave, vb)
		else:
			plot_pdf2d(args[0], nosave, vb)
	## Plot all files
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_CAR_U*"+srchstr+"*.npy"))
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			if slices:
				plot_pdf1d(histfile, nosave, vb)
			else:
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
	try:
		P = filename_par(histfile, "_P")
	except:
		P = 0.0
	
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
	
	## For comparison with Nik++16
	rho = rho[:,::-1]
	
	## ------------------------------------------------------------------------
	
	## Plotting
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	fig.canvas.set_window_title("2D PDF")
	lvls = 20
	
	plt.set_cmap("Greys")#Greys coolwarm
	
	
	## Plot density
	cax = ax.contourf(x, y, rho.T, lvls)
#	cbar = fig.colorbar(cax,)
#	cbar.locator = MaxNLocator(nbins=5); cbar.update_ticks()
	
	## Indicate bulk
	yfine = np.linspace(y[0],y[-1],1000)
	ax.scatter(+R-S*np.sin(2*np.pi*yfine/T), yfine, c="k", s=1)
	
	ax.set_xlim(xbins[0],xbins[-1])
	ax.set_ylim(ybins[0],ybins[-1])
	
	ax.set_xlabel(r"$x/\lambda$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$y/\lambda$", fontsize=fs["fsa"])
	
	ax.xaxis.set_major_locator(MaxNLocator(5))
	
	ax.grid()
	
	## ------------------------------------------------------------------------
	## Potential inset
	
	## Plot potential in 3D
#	left, bottom, width, height = [0.44, 0.16, 0.30, 0.30]	## For lower right
#	axin = fig.add_axes([left, bottom, width, height], projection="3d")
#	Rschem, Sschem, Tschem = (2.0,1.0,1.0)
#	plot_U3D_ulin(axin, Rschem, Sschem, Tschem)
	## Plot potential in 2D
	try:
		cbar
		left, bottom, width, height = [0.47, 0.16, 0.25, 0.25]	## For lower right
	except:
		left, bottom, width, height = [0.57, 0.16, 0.30, 0.30]
	axin = fig.add_axes([left, bottom, width, height])
	Rschem, Sschem, Tschem = (2.0,1.0,1.0)
	plot_U2D_ulin(axin, Rschem, Sschem, Tschem)
	
	axin.set_axis_bgcolor(plt.get_cmap()(0.00))
	axin.patch.set_facecolor("None")
		
	## ------------------------------------------------------------------------
	
	title = r"PDF projections. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f, P=%.1f\pi$"%(a,R,S,T,P)
	
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFxy2d"+os.path.basename(histfile)[4:-4]
		plotfile += "."+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)

	return
	
##=============================================================================
def plot_pdf1d(histfile, nosave, vb):
	"""
	Read in data for a single file and plot a few 1D PDF slices.
	"""
	me = me0+".plot_pdf1D: "
	t0 = time.time()

	##-------------------------------------------------------------------------
	
	## Get pars from filename
	
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T")
	try:
		P = filename_par(histfile, "_P")
	except:
		P = 0.0
	
	##-------------------------------------------------------------------------
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	ybins = bins["ybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	y = 0.5*(ybins[1:]+ybins[:-1])
	
	##-------------------------------------------------------------------------
	
	## Histogram / density
	H = np.load(histfile)
	rho = H / (H.sum() * (x[1]-x[0])*(y[1]-y[0]))
	
	## For comparison with Nik++16
	rho = rho[:,::-1]
	
	## ------------------------------------------------------------------------
	
	## Plotting
		
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	fig.canvas.set_window_title("1D PDF")
	
	## Slices to plot
	idxs = np.linspace(y.size/4,y.size*3/4,11)
	labs = [r"$"+str(float(i)/y.size)+"$" for i in idxs]
	
	## Plot density and wall
	for i, idx in enumerate(idxs):
		sp.ndimage.gaussian_filter1d(rho[:,idx],1.0,order=0,output=rho[:,idx])
		off = 0.0*S*np.sin(idxs[i]/y.size*T*2*np.pi)
		ax.plot(x+off, rho[:,idx], label=labs[i])
		ax.axvline(+R-S*np.sin(2*np.pi*y[idx]/T)+off, c=ax.lines[-1].get_color(),ls="--")
	
	ax.set_xlim(xbins[0],xbins[-1])
	
	ax.set_xlabel(r"$x/\lambda$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$n(x,y^\ast)$", fontsize=fs["fsa"])
	
	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(7))
	ax.grid()
	leg = ax.legend(loc="upper left",ncol=1)
	leg.set_title(r"$y^\ast/\lambda$", prop={"size":fs["fsl"]})
	# leg.get_frame().set_alpha(0.7)
	
	## ------------------------------------------------------------------------
	
	title = r"PDF slices. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f, P=%.1f\pi$"%(a,R,S,T,P)
#	fig.suptitle(title, fontsize=fs["fst"])
	
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFxy1d"+os.path.basename(histfile)[4:-4]
		plotfile += "."+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)

	return
	
	
##=============================================================================
def plot_pdf1d_intx(histfile, nosave, vb):
	"""
	Read in data for a single file and plot the density integrated from the wall to infinity
	as a function of y.
	"""
	me = me0+".plot_pdf1D_intx: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Get pars from filename
	
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T")
	try:
		P = filename_par(histfile, "_P")
	except:
		P = 0.0
	
	##-------------------------------------------------------------------------
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	ybins = bins["ybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	y = 0.5*(ybins[1:]+ybins[:-1])
	
	##-------------------------------------------------------------------------
	
	## Histogram / density
	H = np.load(histfile)
	rho = H / (H.sum() * (x[1]-x[0])*(y[1]-y[0]))
	
	## For comparison with Nik++16
	rho = rho[:,::-1]
	
	## Integrate rho over x IN WALL REGION
	Qy = np.zeros(y.size)
	wind = +R-S*np.sin(2*np.pi*y/T)	### HARD CODED
	for i,yi in enumerate(y):
		Qy[i] = np.trapz(rho[wind[i]:,i],x[wind[i]:],axis=0)
	
	## ------------------------------------------------------------------------
	
	## Plotting
		
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	fig.canvas.set_window_title("1D PDF integrated along x")
		
	## Plot density and wall
	ax.plot(y, Qy)
	
	## Indicate inflexion point
	ax.axvspan(y[0],0.5*T, color="g",alpha=0.2)
	
	ax.set_xlabel(r"$y$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$Q_x(y)$", fontsize=fs["fsa"])
	
	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(7))
	ax.grid()
	ax.legend(loc="upper left")
	
	## ------------------------------------------------------------------------
	
	title = r"PDF slice integrated over x in wall region. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f, P=%.1f\pi$"%(a,R,S,T,P)
	fig.suptitle(title, fontsize=fs["fst"])
	
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFxy1dintx"+os.path.basename(histfile)[4:-4]
		plotfile += "."+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)

	return

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
