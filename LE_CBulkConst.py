me0 = "LE_CBulkConst"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import pyplot as plt

from LE_Utils import filename_pars, filename_par
from LE_Utils import fs
fsa,fsl,fst = fs
from LE_CSim import force_dlin, force_clin, force_mlin

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

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
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
	nosave = opt.nosave
	verbose = opt.verbose
		
	if os.path.isfile(args[0]):
		plot_file(args[0], nosave, verbose)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+srchstr+"*.npy"))
		assert len(filelist)>1, me+"Check directory."
		if verbose: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_file(histfile, nosave, verbose)
			plt.close()
	elif os.path.isdir(args[0]):
		plot_dir(args[0], nosave, srchstr, verbose)
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
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
	Casimir = "_CL_" in histfile

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
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
#	Qx_WN = np.exp(-U)/np.trapz(np.exp(-U),x)
	
	##-------------------------------------------------------------------------
	
	## PLOT
	fig, ax = plt.subplots(1,1, figsize=(10,10))
	
	## Data
	ax.plot(x, Q/Q.max(),   label=r"$Q(x)$")
	ax.plot(x, BC/BC.max(), label=r"Bulk constant", lw=2)
	ax.plot(x, ex2/ex2.max(), label=r"$\langle\eta_x^2\rangle Q$")
	
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", label=r"$U(x)$")	
		
	## Indicate bulk region
	ax.axvspan(S,R, color="yellow",alpha=0.2)
	ax.axvline(S, c="k",lw=2);	ax.axvline(R, c="k",lw=2)
	if Casimir:
		ax.axvspan(0.0,T, color="yellow",alpha=0.2)
		ax.axvline(T, c="k",lw=2)
		
	##-------------------------------------------------------------------------
	
	## ATTRIBUTES
	
	ax.set_xlim(left=x[0],right=x[-1])

	ax.set_xlabel("$x$",fontsize=fsa)
	ax.set_ylabel("Rescaled variable",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl).get_frame().set_alpha(0.5)
	ax.set_title(r"Bulk Constant. $\alpha=%.1f, R=%.1f, S=%.1f$."%(a,R,S),fontsize=fst)
	
	## SAVE
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+".jpg"
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
	
	##-------------------------------------------------------------------------
	
	return plotfile
	
##=============================================================================
def plot_dir(histdir, nosave, srchstr, vb):
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
	
		Casimir = "_CL_" in histfile

		## Get pars from filename
		A[i] = filename_par(histfile, "_a")
		R = filename_par(histfile, "_R")
		S = filename_par(histfile, "_S")
		T = filename_par(histfile, "_T") if Casimir else -S
		
		## Calculate BC
		x, Qx, BC = bulk_const(histfile)[:3]
		
		## Wall indices
		Rind, Sind, Tind = np.abs(x-R).argmin(), np.abs(x-S).argmin(), np.abs(x-T).argmin()
		STind = (Tind+Sind)/2
		
		##---------------------------------------------------------------------
		## Calculate pressure from BC
		
		BCsr = BC[Sind:Rind+1].mean()
		BCts = BC[STind]
		if Casimir:
			BC0t = BC[0:Tind+1].mean()
		
		pR[i] = A[i] * BCsr
		pS[i] = A[i] * (BCsr - BCts)
		if Casimir:
			pT[i] = A[i] * (BC0t - BCts)
		
		##---------------------------------------------------------------------
		## Calculate pressure from integral
		
		## Choose force
		if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
		elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
		elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
	
		## Calculate integral pressure
		PR[i] = -sp.integrate.trapz(fx[Rind:]*Qx[Rind:], x[Rind:])
		PS[i] = +sp.integrate.trapz(fx[STind:Sind]*Qx[STind:Sind], x[STind:Sind])
		PT[i] = -sp.integrate.trapz(fx[Tind:STind]*Qx[Tind:STind], x[Tind:STind])
		
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	pR, pS, pT = pR[srtidx], pS[srtidx], pT[srtidx]
	PR, PS, PT = PR[srtidx], PS[srtidx], PT[srtidx]
	
	##-------------------------------------------------------------------------
	
	## Calculate white noise PDF and pressure
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U)/np.trapz(np.exp(-U),x)
	PR_WN = -sp.integrate.trapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:])
	PS_WN = +sp.integrate.trapz(fx[STind:Sind]*Qx_WN[STind:Sind], x[STind:Sind])
	PT_WN = -sp.integrate.trapz(fx[Tind:STind]*Qx_WN[Tind:STind], x[Tind:STind])
	
	## Normalise
	pR /= PR_WN; pS /= PS_WN; pT /= PT_WN
	PR /= PR_WN; PS /= PS_WN; PT /= PT_WN
	
	##-------------------------------------------------------------------------
	
	## PLOT DATA
	
	fig, ax = plt.subplots(1,1, figsize=(10,10))
	
	lpR = ax.plot(A, pR, "o-", label=r"BC pR")
	lpS = ax.plot(A, pS, "o-", label=r"BC pS")
	lpT = ax.plot(A, pT, "o-", label=r"BC pT")
	
	lPR = ax.plot(A, pR, lpR[0].get_color()+"v--", lw=2, label=r"Int PR")
	lPS = ax.plot(A, pS, lpS[0].get_color()+"v--", lw=2, label=r"Int PS")
	lPT = ax.plot(A, pT, lpT[0].get_color()+"v--", lw=2, label=r"Int PT")
		
	##-------------------------------------------------------------------------
	
	## ACCOUTREMENTS
#	ax.set_xscale("log")
#	ax.set_yscale("log")
	ax.set_xlim(0,A[-1])
	
	ax.set_xlabel(r"$\alpha$",fontsize=fsa)
	ax.set_ylabel(r"$P$",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best", fontsize=fsl).get_frame().set_alpha(0.5)
	title = "Pressure normalised by WN result. $R=%.1f, S=%.1f, T=%.1f.$"%(R,S,T) if Casimir\
			else "Pressure normalised by WN result. $R=%.1f, S=%.1f.$"%(R,S)
	ax.set_title(title,fontsize=fst)
	
	## SAVING
	plotfile = histdir+"/QEe2_Pa_R%.1f_S%.1f_T%.1f.jpg"%(R,S,T) if Casimir\
				else histdir+"/QEe2_Pa_R%.1f_S%.1f.jpg"%(R,S)
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
	me = me0+",bulk_const: "

	Casimir = "_CL_" in histfile
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T") if Casimir else -S
		
	## Space and load histogram
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	exbins = bins["exbins"]
	eybins = bins["eybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	ex = 0.5*(exbins[1:]+exbins[:-1])
	ey = 0.5*(eybins[1:]+eybins[:-1])
	
	## Wall indices
	Rind, Sind, Tind = np.abs(x-R).argmin(), np.abs(x-S).argmin(), np.abs(x-S).argmin()

	## --------------------------------------------------------------------
	
	## Load histogram
	H = np.load(histfile)
	rhoxex = H.sum(axis=2) / (H.sum()*(x[1]-x[0])*(ex[1]-ex[0])*(ey[1]-ey[0]))
	## Spatial density
	Q = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))
	
	## Bulk constant as a continuous integral. A function of x. 
	BC = np.trapz(rhoxex*ex*ex, ex)
				
	return x, Q, BC
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
