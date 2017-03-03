me0 = "LE_CBulkConst"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

from LE_Utils import filename_par, fs, set_mplrc
from LE_CSim import force_dlin, force_clin, force_mlin, force_nlin

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## Plot defaults
set_mplrc(fs)

## ============================================================================

def main():
	"""
	Plot the bulk constant in CARTESIAN coordinates.
	As a function of x for a single file, or plotting its mean value against alpha.
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
	parser.add_option('-v','--verbose',
		dest="vb", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
	logplot = opt.logplot
	nosave = opt.nosave
	vb = opt.vb
		
	if os.path.isfile(args[0]):
		plot_file(args[0], nosave, vb)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+srchstr+"*.npy"))
		assert len(filelist)>1, me+"Check directory."
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_file(histfile, nosave, vb)
			plt.close()
	elif os.path.isdir(args[0]):
		plot_dir(args[0], srchstr, logplot, nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile, nosave, vb):
	"""
	"""
	me = me0+".plot_file: "
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histfile, me+"Functional only for Cartesian geometry."
	Casimir = "_CL_" in histfile or "_ML_" in histfile

	## Get pars from filename
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T") if Casimir else -S
	
	## Calculate quantities
	x, Q, BC = bulk_const(histfile)[:3]
	ex2 = BC/(Q+(Q==0.0))
	
	##-------------------------------------------------------------------------
		
	## Potential
	if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
	elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
	elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
	elif "_NL_" in histfile:	fx = force_nlin([x,0],R,S)[0]
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	
	##-------------------------------------------------------------------------
	
	## Smooth
	sp.ndimage.gaussian_filter1d(Q,1.0,order=0,output=Q)
	sp.ndimage.gaussian_filter1d(BC,1.0,order=0,output=BC)
	sp.ndimage.gaussian_filter1d(ex2,1.0,order=0,output=ex2)
	
	##-------------------------------------------------------------------------
	
	## PLOT
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	## Data
	ax.plot(x, Q/Q.max(),   label=r"$n(x)$",lw=2)
	ax.plot(x, ex2/ex2.max(), label=r"$\langle\eta_x^2\rangle(x)$",lw=2)
	ax.plot(x, BC/BC.max(), label=r"$\langle\eta_x^2\rangle \cdot n$",lw=2)
	
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", label=r"$U(x)$")	
		
	## Indicate bulk region
	if "_DL_" in histfile:
		ax.axvspan(S,R, color="yellow",alpha=0.2)
		ax.axvline(S, c="k",lw=2);	ax.axvline(R, c="k",lw=2)
	elif "_ML_" in histfile:
		ax.axvspan(S,R, color="yellow",alpha=0.2)
		ax.axvspan(-R,T, color="yellow",alpha=0.2)
		ax.axvline(S, c="k",lw=2);	ax.axvline(R, c="k",lw=2)
		ax.axvline(T, c="k",lw=2);	ax.axvline(-R, c="k",lw=2)
	elif "_CL_" in histfile:
		ax.axvspan(S,R, color="yellow",alpha=0.2)
		ax.axvspan(0,T, color="yellow",alpha=0.2)
		ax.axvline(S, c="k",lw=2);	ax.axvline(R, c="k",lw=2)
		ax.axvline(T, c="k",lw=2);	ax.axvline(-R, c="k",lw=2)
		
	##-------------------------------------------------------------------------
	
	## ATTRIBUTES
	
	ax.set_xlim(left=x[0],right=x[-1])
	ax.xaxis.set_major_locator(NullLocator())
	ax.yaxis.set_major_locator(NullLocator())

	ax.set_xlabel("$x$",fontsize=fs["fsa"])
	ax.set_ylabel("Rescaled variable",fontsize=fs["fsa"])
	ax.grid()
	legloc = [0.35,0.25] if "_ML_" in histfile else [0.32,0.67]
	ax.legend(loc=legloc,fontsize=fs["fsl"]).get_frame().set_alpha(0.8)
	title = r"Bulk Constant. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f$."%(a,R,S,T) if T>=0.0\
			else r"Bulk Constant. $\alpha=%.1f, R=%.1f, S=%.1f$."%(a,R,S)
#	fig.suptitle(title,fontsize=fs["fst"])
	
	## SAVE
#	ax.set_ylim(top=BC.max())
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+"."+fs["saveext"]
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
	
	##-------------------------------------------------------------------------
	
	return plotfile
	
##=============================================================================
def plot_dir(histdir, srchstr, logplot, nosave, vb):
	"""
	For each file in directory, calculate the pressure in both ways for all walls
	(where applicable) and plot against alpha.
	"""
	me = me0+".plot_dir: "
	
	filelist = np.sort(glob.glob(histdir+"/BHIS_CAR_*"+srchstr+"*.npy"))
	numfiles = filelist.size
	if vb: print me+"Found",numfiles,"files."
	
	## Initialise arrays
	A, pR, pS, pT, PR, PS, PT = np.zeros([7,numfiles])	
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
	
		Casimir = "_CL_" in histfile or "_ML_" in histfile or "_NL_" in histfile

		## Get pars from filename
		A[i] = filename_par(histfile, "_a")
		R = filename_par(histfile, "_R")
		S = filename_par(histfile, "_S")
		T = filename_par(histfile, "_T") if Casimir else -S
		
		## Calculate BC
		x, Qx, BC = bulk_const(histfile)[:3]
		
		## Wall indices
		Rind, Sind, Tind = np.abs(x-R).argmin(), np.abs(x-S).argmin(), np.abs(x-T).argmin()
		STind = 0 if "_DL_" in histfile else (Tind+Sind)/2
		
		##---------------------------------------------------------------------
		## Calculate pressure from BC
		
		if "_DL_" in histfile:	
			BCsr = BC[Sind:Rind+1].mean()
			pR[i] = A[i] * BCsr
			pS[i] = A[i] * BCsr
			
		elif "_CL_" in histfile:
			BCsr = BC[Sind:Rind+1].mean()
			BCts = BC[STind]
			BC0t = BC[0:Tind+1].mean()
			pR[i] = A[i] * BCsr
			pS[i] = A[i] * (BCsr - BCts)
			pT[i] = A[i] * (BC0t - BCts)
			
		elif "_ML_" in histfile:
			BCsr = BC[Sind:Rind+1].mean()
			BCts = BC[STind]
			BCrt = BC[x.size-Rind:Tind+1].mean()
			pR[i] = A[i] * BCsr
			pS[i] = A[i] * (-BCsr + BCts)
			pT[i] = A[i] * (-BCrt + BCts)
			
		elif "_NL_" in histfile:
			BCr = BC[Rind]
			BCs = BC[Sind]
			BCmr = BC[x.size-Rind]
			pR[i] = A[i] * BCr
			pS[i] = A[i] * (BCs - BCr)
			pT[i] = A[i] * (BCs - BCmr)
		
		##---------------------------------------------------------------------
		## Calculate pressure from integral
		
		## Choose force
		if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
		elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
		elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
		elif "_NL_" in histfile:	fx = force_nlin([x,0],R,S)[0]
	
		## Calculate integral pressure
		PR[i] = -sp.integrate.trapz(fx[Rind:]*Qx[Rind:], x[Rind:])
		PS[i] = +sp.integrate.trapz(fx[STind:Sind]*Qx[STind:Sind], x[STind:Sind])
		PT[i] = -sp.integrate.trapz(fx[Tind:STind]*Qx[Tind:STind], x[Tind:STind])
		
		##---------------------------------------------------------------------
		
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	pR, pS, pT = pR[srtidx], pS[srtidx], pT[srtidx]
	PR, PS, PT = PR[srtidx], PS[srtidx], PT[srtidx]
	
	##-------------------------------------------------------------------------
	
	## Calculate white noise PDF and pressure -- assuming alpha is only varying parameter
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U) / np.trapz(np.exp(-U),x)
	
	PR_WN = -sp.integrate.trapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:])
	PS_WN = +sp.integrate.trapz(fx[STind:Sind]*Qx_WN[STind:Sind], x[STind:Sind])
	PT_WN = -sp.integrate.trapz(fx[Tind:STind]*Qx_WN[Tind:STind], x[Tind:STind])
	
	## Normalise
	pR /= PR_WN; pS /= PS_WN; pT /= PT_WN
	PR /= PR_WN; PS /= PS_WN; PT /= PT_WN
	
	##-------------------------------------------------------------------------
	
	## Add a=0 point
	if 0.0 not in A:
		nlin = np.unique(S).size
		A = np.hstack([[0.0]*nlin,A])
		pR = np.hstack([[1.0]*nlin,pR])
		pS = np.hstack([[1.0]*nlin,pS])
		PR = np.hstack([[1.0]*nlin,PR])
		PS = np.hstack([[1.0]*nlin,PS])
		
	##-------------------------------------------------------------------------
	
	## PLOT DATA
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	sty = ["-","--",":"]
	
	A += int(logplot)
	
	"""
	lpR = ax.plot(A, pR, "o"+sty[0], label=r"BC pR")
	lpS = ax.plot(A, pS, "o"+sty[1], c=ax.lines[-1].get_color(), label=r"BC pS")
	if Casimir:	
		lpT = ax.plot(A, pT, "o"+sty[2], c=ax.lines[-1].get_color(), label=r"BC pT")
	
	ax.plot(A, PR, "v"+sty[0], label=r"Int PR")
	ax.plot(A, PS, "v"+sty[1], c=ax.lines[-1].get_color(), label=r"Int PS")
	if Casimir:	
		ax.plot(A, PT, "v"+sty[2], c=ax.lines[-1].get_color(), label=r"Int PT")
	"""
	lpR = ax.plot(A, 0.5*(pR+pS), "o--", label=r"$\alpha\left<\eta^2\right>n(x)|^{\rm bulk}$")
	ax.plot(A, 0.5*(PR+PS), "v--", label=r"$-\int f(x)n(x) {\rm d}x$")
		
	##-------------------------------------------------------------------------
	
	## ACCOUTREMENTS
	
	if logplot:
		ax.set_xscale("log"); ax.set_yscale("log")
		xlim = (ax.get_xlim()[0],A[-1])
		xlabel = r"$1+\alpha$"
	else:
		xlim = (0.0,A[-1])
		xlabel = r"$\alpha$"
		
	ax.set_xlim(xlim)
	ax.set_ylim(1e-1,1e+1)
	ax.set_xlabel(xlabel,fontsize=fs["fsa"])
	ax.set_ylabel(r"$P(\alpha)/P^{\rm passive}$",fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="best", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	title = "Pressure normalised by WN result. $R=%.1f, S=%.1f, T=%.1f.$"%(R,S,T) if T>=0.0\
			else "Pressure normalised by WN result. $R=%.1f, S=%.1f.$"%(R,S)
#	fig.suptitle(title,fontsize=fs["fst"])
	
	## SAVING
	plotfile = histdir+"/QEe2_Pa_R%.1f_S%.1f_T%.1f"%(R,S,T) if T>=0.0\
				else histdir+"/QEe2_Pa_R%.1f_S%.1f"%(R,S)
	plotfile += "_loglog"*logplot+"."+fs["saveext"]
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
		
	return plotfile
	
##=============================================================================
##=============================================================================

def bulk_const(histfile):
	"""
	Calculate various quantities pertaining to the moment-pressure calculation.
	"""
	me = me0+".bulk_const: "
		
	## Space and load histogram
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	exbins = bins["exbins"]
	eybins = bins["eybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	ex = 0.5*(exbins[1:]+exbins[:-1])
	ey = 0.5*(eybins[1:]+eybins[:-1])

	## --------------------------------------------------------------------
	
	## Load histogram
	H = np.load(histfile)
	rhoxex = H.sum(axis=2) / (H.sum()*(x[1]-x[0])*(ex[1]-ex[0]))
	
	## Spatial density
	Q = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))
	
	## Bulk constant as a function of x: <etax^2>Q
	BC = np.trapz(rhoxex*ex*ex, ex, axis=1)
				
	return x, Q, BC
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
