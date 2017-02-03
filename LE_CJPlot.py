me0 = "LE_CJPlot"

import numpy as np
import scipy as sp
import os, optparse, glob, time

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib.ticker import MaxNLocator, NullLocator
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
	Plot the marginalised densities Q(x), qx(etax) and  qy(etay).
	Adapted from LE_PDFre.py.
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

	## Plot file
	assert os.path.isfile(args[0])
	if os.path.basename(args[0])[:8] =="BHIS_CAR":
		plot_current_1d(args[0], nosave, vb)
	if os.path.basename(args[0])[:11]=="CURR_CAR_UL":
		plot_current_2d(args[0], nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_current_1d(histfile, nosave, vb):
	"""
	"""
	me = me0+".plot_current_1d: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Filename pars
	
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	
	##-------------------------------------------------------------------------
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	exbins = bins["exbins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	etax = 0.5*(exbins[1:]+exbins[:-1])
	X, ETAX = np.meshgrid(x,etax, indexing="ij")
	
	## Wall indices
	Rind, Sind = np.abs(x-R).argmin(), np.abs(x-S).argmin()
	
	##-------------------------------------------------------------------------
	
	## Force
	if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
	elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
	elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
	else: raise IOError, me+"Force not recognised."
	
	F = fx.repeat(etax.size).reshape([x.size,etax.size])
	
	##-------------------------------------------------------------------------
	
	## Histogram
	H = np.load(histfile)
	rho = H.sum(axis=2) / (H.sum() * (x[1]-x[0])*(etax[1]-etax[0]))
	
	## Currents
	Jx = (F + ETAX)*rho
	Jy = -1/a*ETAX*rho - 1/(a*a)*np.gradient(rho,etax[1]-etax[0],axis=1)
	Vx, Vy = Jx/rho, Jy/rho
	
	##-------------------------------------------------------------------------
	
	## SMOOTHING
	
	Vy = sp.ndimage.gaussian_filter(Vy, 1.0, order=0)
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
				
	plt.rcParams["image.cmap"] = "Greys"
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	fig.canvas.set_window_title("Current in x-eta")
	
	##-------------------------------------------------------------------------
	
	## Data
	ax.contourf(x, etax, rho.T)
	sx, se = 30, 5
	ax.quiver(x[::sx], etax[::se], Vx.T[::se,::sx], Vy.T[::se,::sx])
	
	
	## Indicate bulk
	ax.axvline(S,c="k",lw=1)
	ax.axvline(R,c="k",lw=1)
	
	## Set number of ticks
	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(4))

	ax.set_xlim(left=x[0],right=x[-1])
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$\eta$", fontsize=fs["fsa"])
	ax.grid()
	# ax.legend(loc="upper right", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
		
	##-------------------------------------------------------------------------
		
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/Jxeta"+os.path.basename(histfile)[4:-4]
		plotfile += "."+fs["saveext"]
		fig.savefig(plotfile, format=fs["saveext"])
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	


##=============================================================================
def plot_current_2d(currfile, nosave, vb):
	"""
	"""
	me = me0+".plot_current_2d: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Filename pars
	
	a = filename_par(currfile, "_a")
	R = filename_par(currfile, "_R")
	S = filename_par(currfile, "_S")
	T = filename_par(currfile, "_T")
	
	##-------------------------------------------------------------------------
		
	## Load data
	
	data = np.load(currfile)
	xbins = data["xbins"]
	ybins = data["ybins"]
	vxbins = data["vxbins"]
	vybins = data["vybins"]
	Hxy = data["Hxy"][:,::-1]
	Vx = data["Vx"][:,::-1]
	Vy = data["Vy"][:,::-1]
	del data
	
	x = 0.5*(xbins[:-1]+xbins[1:])
	y = 0.5*(ybins[:-1]+ybins[1:])
	vx = 0.5*(vxbins[:-1]+vxbins[1:])
	vy = 0.5*(vybins[:-1]+vybins[1:])
		
	##-------------------------------------------------------------------------
	
	## SMOOTHING
	
#	Vy = sp.ndimage.gaussian_filter(Vy, 1.0, order=0)
	
	zx, zy = 0.2, 1.0
	Vx = sp.ndimage.interpolation.zoom(Vx, (zx,zy), mode="nearest", cval=0.0)
	Vy = sp.ndimage.interpolation.zoom(Vy, (zx,zy), mode="nearest", cval=0.0)
	Vy[:,0]  = 0.0
	Vy[:,-1] = 0.0
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
				
	plt.rcParams["image.cmap"] = "Greys"
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	fig.canvas.set_window_title("Current in x-eta")
	
	##-------------------------------------------------------------------------
	
	## Data
	ax.contourf(x, y, Hxy.T)
	sx, sy = 1, 1
	ax.quiver(x[::int(1/zx)], y[::int(1/zy)], Vx.T[::sy,::sx], Vy.T[::sy,::sx])
	
	## Indicate wall
	yfine = np.linspace(0,T,1000)
	ax.scatter(+R-S*np.sin(2*np.pi*yfine/T), yfine, c="k", s=1)
	
	## Set number of ticks
	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(4))

	ax.set_xlim([xbins[0],xbins[-1]])
	ax.set_ylim([ybins[0],ybins[-1]])
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	ax.grid()
	# ax.legend(loc="upper right", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
		
	##-------------------------------------------------------------------------
		
	if not nosave:
		plotfile = os.path.dirname(currfile)+"/Jxy"+os.path.basename(currfile)[4:-4]
		plotfile += "."+fs["saveext"]
		fig.savefig(plotfile, format=fs["saveext"])
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
