me0 = "LE_C2Pressure"

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

from LE_CSim import force_ulin, potential_ulin
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
		dest="srchstr", default="", type="str")
	parser.add_option('--logplot',
		dest="logplot", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('--noread',
		dest="noread", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
	logplot = opt.logplot
	nosave = opt.nosave
	noread = opt.noread
	vb = opt.verbose
	
	assert "_CAR_" in args[0], me+"Functional only for Cartesian geometry."
	assert "_UL_" in args[0], me+"Functional only for ulin ftype."
	
	## Plot directory
	if os.path.isdir(args[0]):
		plot_pressure_dir(args[0], srchstr, logplot, nosave, noread, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def calc_pressure_dir(histdir, srchstr, noread, vb):
	"""
	Calculate the pressure for all files in directory matching string.
	"""
	me = me0+".calc_pressure_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## File discovery
	filelist = np.sort(glob.glob(histdir+"/BHIS_CAR_UL_*"+srchstr+"*.npy"))
	numfiles = len(filelist)
	assert numfiles>1, me+"Check input directory."
	if vb: print me+"found",numfiles,"files"

	##-------------------------------------------------------------------------
	
	A, R, S, T, Pt, Pt_WN = np.zeros([6,numfiles])
	Py = []
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
		
		ti = time.time()
		
		## Assuming R, S, T are same for all files
		A[i] = filename_par(histfile, "_a")
		R[i] = filename_par(histfile, "_R")
		S[i] = filename_par(histfile, "_S")
		T[i] = filename_par(histfile, "_T")
			
		## Space
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		xbins = bins["xbins"]
		ybins = bins["ybins"]
		x = 0.5*(xbins[1:]+xbins[:-1])
		y = 0.5*(ybins[1:]+ybins[:-1])
		
		## Wall indices
		
		##-------------------------------------------------------------------------
		
		## Histogram
		H = np.load(histfile)
		## Spatial density
		Qxy = H / (H.sum()*(x[1]-x[0])*(y[1]-y[0]))
		
		##-------------------------------------------------------------------------
		
		## Force array
		fxy = np.array([force_ulin([xi,yi],R[i],S[i],T[i]) for xi in x for yi in y]).reshape((x.size,y.size,2))
		fxy = np.rollaxis(fxy,2,0)
		
		## Calculate integral pressure for full period in y
		Pt[i] = -np.trapz(np.trapz(fxy[0]*Qxy, y, axis=1), x, axis=0)
		
		if vb: print me+"a=%.1f:\tPressure calculation %.2g seconds"%(A[i],time.time()-ti)
		
		## Potential
		U = potential_ulin(x,y,R[i],S[i],T[i])
		Qxy_WN = np.exp(-U)
		Qxy_WN /= np.trapz(np.trapz(Qxy_WN, y, axis=1), x, axis=0)
		
		## WN pressure
		Pt_WN[i] = -np.trapz(np.trapz(fxy[0]*Qxy_WN, y, axis=1), x, axis=0)
		
	##-------------------------------------------------------------------------
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	R, S, T = R[srtidx], S[srtidx], T[srtidx]
	Pt = Pt[srtidx]
	Pt_WN = Pt_WN[srtidx]
	
	## Normalise
	Pt /= Pt_WN + (Pt_WN==0)
		
	##-------------------------------------------------------------------------
		
	## SAVING
	if not noread:
		pressfile = histdir+"/PRESS_"+srchstr+".npz"
		np.savez(pressfile, A=A, R=R, S=S, T=T, Pt=Pt, Pt_WN=Pt_WN)
		if vb:
			print me+"Calculations saved to",pressfile
			print me+"Calculation time %.1f seconds."%(time.time()-t0)

	return {"A":A, "R":R, "S":S, "T":T, "Pt":Pt, "Pt_WN":Pt_WN}
		

##=============================================================================
def plot_pressure_dir(histdir, srchstr, logplot, nosave, noread, vb):
	"""
	Plot the pressure for all files in directory matching string.
	"""
	me = me0+".plot_pressure_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh
		
	try:
		assert noread == False
		pressdata = np.load(histdir+"/PRESS_"+srchstr+".npz")
		print me+"Pressure data file found:",histdir+"/PRESS_"+srchstr+".npz"
	except (IOError, AssertionError):
		print me+"No pressure data found. Calculating from histfiles."
		pressdata = calc_pressure_dir(histdir, srchstr, noread, vb)
		
	A = pressdata["A"]
	R = pressdata["R"]
	S = pressdata["S"]
	T = pressdata["T"]
	Pt = pressdata["Pt"]
	Pt_WN = pressdata["Pt_WN"]
	del pressdata
		
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	t0 = time.time()
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	A += int(logplot)
	
	##-------------------------------------------------------------------------
	
	## Hold R fixed and vary S and T
	if np.unique(R).size==1:
		plotfile = histdir+"/PAS_R%.1f."%(R[0])+fs["saveext"]
		title = r"Pressure as a function of $\alpha$ for $R=%.1f$"%(R[0])
		for Si in np.unique(S):
			for Ti in np.unique(T):
				idx = (S==Si)*(T==Ti)
				ax.plot(A[idx], Pt[idx], "o-", label=r"$S=%.1f, T=%.1f$"%(Si,Ti))
			
	##-------------------------------------------------------------------------
	
	## Plot appearance
			
	if logplot:
		ax.set_xscale("log"); ax.set_yscale("log")
		ax.set_xlim((ax.get_xlim()[0],A[-1]))
		ax.set_ylim(0.1,10.0)
		xlabel = r"$1+\alpha$"
		plotfile = plotfile[:-4]+"_loglog."+fs["saveext"]
	else:
		ax.set_xlim((0.0,A[-1]))
		ax.set_ylim(bottom=0.0,top=max(ax.get_ylim()[1],1.0))
		xlabel = r"$\alpha$"
	
	ax.set_xlabel(xlabel, fontsize=fs["fsa"])
	ax.set_ylabel(r"$P_{\rm tot}(\alpha)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="best", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	fig.suptitle(title, fontsize=fs["fst"])
	
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Plotting time %.1f seconds."%(time.time()-t0)
	
	return
		

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
