me0 = "LE_CPressure"

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
		dest="srchstr", default="", type="str")
	parser.add_option('--logplot',
		dest="logplot", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
	logplot = opt.logplot
	nosave = opt.nosave
	vb = opt.verbose
	
	## Plot file
	if os.path.isfile(args[0]):
		plot_pressure_file(args[0], nosave, vb)
	## Plot all files
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_CAR_*"+srchstr+"*.npy"))
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pressure_file(histfile, nosave, vb)
			plt.close()
	## Plot directory
	elif os.path.isdir(args[0]):
		plot_pressure_dir(args[0], srchstr, logplot, nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_pressure_file(histfile, nosave, vb):
	"""
	Plot spatial PDF Q(x) and spatially-varying pressure P(x).
	"""
	me = me0+".plot_pressure_file: "
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histfile, me+"Functional only for Cartesian geometry."
	Casimir = "_CL_" in histfile or "_ML_" in histfile or "_NL_" in histfile

	##-------------------------------------------------------------------------
	
	## Filename parameters
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	try: T = filename_par(histfile, "_T")
	except ValueError: T = -S
			
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
		
	## Wall indices
	Rind, Sind, Tind = np.abs(x-R).argmin(), np.abs(x-S).argmin(), np.abs(x-T).argmin()
	STind = (Sind+Tind)/2
	
	if "_NL_" in histfile:
		STind = Sind
		Sind = Rind
		Tind = x.size-Rind
		
	##-------------------------------------------------------------------------
	
	## Histogram
	H = np.load(histfile)
	## Spatial density
	Qx = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))
	
	##-------------------------------------------------------------------------
	
	## Choose force
	if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
	elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
	elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
	elif "_NL_" in histfile:	fx = force_nlin([x,0],R,S)[0]
	else: raise IOError, me+"Force not recognised."
	
	## Calculate integral pressure
	PR = -sp.integrate.cumtrapz(fx[Rind:]*Qx[Rind:], x[Rind:], initial=0.0)
	PS = -sp.integrate.cumtrapz(fx[STind:Sind+1]*Qx[STind:Sind+1], x[STind:Sind+1], initial=0.0); PS -= PS[-1]
	if Casimir:
		PT = -sp.integrate.cumtrapz(fx[Tind:STind+1]*Qx[Tind:STind+1], x[Tind:STind+1], initial=0.0)
	
	if x[0]<0:
		R2ind = x.size-Rind
		PR2 = -sp.integrate.cumtrapz(fx[:R2ind]*Qx[:R2ind], x[:R2ind], initial=0.0); PR2 -= PR2[-1]
			
	##-------------------------------------------------------------------------
	
	## Potential and WN
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U) / np.trapz(np.exp(-U), x)
	
	## WN pressure
	PR_WN = -sp.integrate.cumtrapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:], initial=0.0)
	PS_WN = -sp.integrate.cumtrapz(fx[STind:Sind+1]*Qx_WN[STind:Sind+1], x[STind:Sind+1], initial=0.0); PS_WN -= PS_WN[-1]
	if Casimir:
		PT_WN = -sp.integrate.cumtrapz(fx[Tind:STind+1]*Qx_WN[Tind:STind+1], x[Tind:STind+1], initial=0.0)
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	fig, axs = plt.subplots(2,1, sharex=True, figsize=fs["figsize"])
	
	if   "_DL_" in histfile:	legloc = "upper right"
	elif "_CL_" in histfile:	legloc = "upper right"
	elif "_ML_" in histfile:	legloc = "upper left"
	elif "_NL_" in histfile:	legloc = "lower left"
	else:						legloc = "best"
	
	## Plot PDF
	ax = axs[0]
	lQ = ax.plot(x, Qx, lw=2, label=r"CN")
	ax.plot(x, Qx_WN, lQ[0].get_color()+":", lw=2, label="WN")
	## Potential
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", lw=2, label=r"$U(x)$")
	
	ax.set_xlim((x[0],x[-1]))	
	ax.set_ylim(bottom=0.0)	
	ax.set_ylabel(r"$Q(x)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc=legloc, fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	
	## Plot pressure
	ax = axs[1]
	lPR = ax.plot(x[Rind:], PR, lw=2, label=r"$P_R$")
	lPS = ax.plot(x[STind:Sind+1], PS, lw=2, label=r"$P_S$")
	if Casimir:
		lPT = ax.plot(x[Tind:STind+1], PT, lw=2, label=r"$P_T$")
	if x[0]<0:
		ax.plot(x[:R2ind], PR2, lPR[0].get_color()+"-", lw=2)
	## WN result
	ax.plot(x[Rind:], PR_WN, lPR[0].get_color()+":", lw=2)
	ax.plot(x[STind:Sind+1], PS_WN, lPS[0].get_color()+":", lw=2)
	if Casimir:
		ax.plot(x[Tind:STind+1], PT_WN, lPT[0].get_color()+":", lw=2)
	if x[0]<0:
		ax.plot(x[:R2ind], PR_WN[::-1], lPR[0].get_color()+":", lw=2)
	## Potential
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", lw=2)#, label=r"$U(x)$")
	
	ax.set_xlim((x[0],x[-1]))	
	ax.set_ylim(bottom=0.0)	
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$P(x)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc=legloc, fontsize=fs["fsl"]).get_frame().set_alpha(0.5)

	##-------------------------------------------------------------------------
	
	fig.tight_layout()
	fig.subplots_adjust(top=0.90)
	title = r"Spatial PDF and Pressure. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f$"%(a,R,S,T) if T>=0.0\
			else r"Spatial PDF and Pressure. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S)
	fig.suptitle(title, fontsize=fs["fst"])
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	return
	
##=============================================================================
def plot_pressure_dir(histdir, srchstr, logplot, nosave, vb):
	"""
	"""
	me = me0+".plot_pressure_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histdir, me+"Functional only for Cartesian geometry."
	Casimir = "_DL_" not in histdir
	
	## File discovery
	filelist = np.sort(glob.glob(histdir+"/BHIS_CAR_*"+srchstr+"*.npy"))
	numfiles = len(filelist)
	assert numfiles>1, me+"Check input directory."
	if vb: print me+"found",numfiles,"files"

	##-------------------------------------------------------------------------
	
	A, PR, PS, PT = np.zeros([4,numfiles])
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
		
		ti = time.time()
		
		## Assuming R, S, T are same for all files
		A[i] = filename_par(histfile, "_a")
		R = filename_par(histfile, "_R")
		S = filename_par(histfile, "_S")
		try: 
			T = filename_par(histfile, "_T")
		except ValueError:
			T = -S
			
		## Space
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		xbins = bins["xbins"]
		x = 0.5*(xbins[1:]+xbins[:-1])
		
		## Wall indices
		Rind, Sind, Tind = np.abs(x-R).argmin(), np.abs(x-S).argmin(), np.abs(x-T).argmin()
		STind = (Sind+Tind)/2
		
		##-------------------------------------------------------------------------
		
		## Histogram
		H = np.load(histfile)
		## Spatial density
		Qx = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))
		
		##-------------------------------------------------------------------------
		
		## Choose force
		if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
		elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
		elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
		elif "_NL_" in histfile:	fx = force_nlin([x,0],R,S)[0]
		else: raise IOError, me+"Force not recognised."
		
		## Calculate integral pressure
		PR[i] = -sp.integrate.trapz(fx[Rind:]*Qx[Rind:], x[Rind:])
		PS[i] = +sp.integrate.trapz(fx[STind:Sind]*Qx[STind:Sind], x[STind:Sind])
		PT[i] = -sp.integrate.trapz(fx[Tind:STind]*Qx[Tind:STind], x[Tind:STind])
		
		if vb: print me+"a=%.1f:\tPressure calculation %.2g seconds"%(A[i],time.time()-ti)
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	PR, PS, PT = PR[srtidx], PS[srtidx], PT[srtidx]
	
	##-------------------------------------------------------------------------
	
	## Potential and WN normalisation
	
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U) / np.trapz(np.exp(-U), x)
	
	PR_WN = -sp.integrate.trapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:])
	PS_WN = +sp.integrate.trapz(fx[STind:Sind]*Qx_WN[STind:Sind], x[STind:Sind])
	if Casimir:
		PT_WN = -sp.integrate.trapz(fx[Tind:STind]*Qx_WN[Tind:STind], x[Tind:STind])

	PR /= PR_WN + (PR_WN==0)
	PS /= PS_WN + (PS_WN==0)
	if Casimir:
		PT /= PT_WN + (PT_WN==0)
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	plotfile = histdir+"/PA_R%.1f_S%.1f_T%.1f."%(R,S,T)+fs["saveext"] if T>=0.0\
				else histdir+"/PA_R%.1f_S%.1f."%(R,S)+fs["saveext"]
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	ax.plot(A, PR, "o-", label=r"$P_R$")
	ax.plot(A, PS, "o-", label=r"$P_S$")
	if Casimir:
		ax.plot(A, PT, "o-", label=r"$P_T$")
		
	if logplot:
		ax.set_xscale("log"); ax.set_yscale("log")
		plotfile += "_loglog"
	else:
		ax.set_ylim(bottom=0.0,top=max(ax.get_ylim()[1],1.0))
	
	ax.set_xlabel(r"$\alpha$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$P(\alpha)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="lower left", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	title = r"Pressure as a function of $\alpha$ for $R=%.1g,S=%.1g,T=%.1g$"%(R,S,T) if T>=0.0\
			else r"Pressure as a function of $\alpha$ for $R=%.1g,S=%.1g$"%(R,S)
	fig.suptitle(title, fontsize=fs["fst"])
	
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
