me0 = "LE_CPressure"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt

from LE_CSim import force_dlin, force_clin
from LE_Utils import filename_par
from LE_Utils import fs
fsa,fsl,fst = fs

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

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
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	srchstr = opt.srchstr
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
		plot_pressure_dir(args[0], srchstr, nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_pressure_file(histfile, nosave, vb):
	"""
	"""
	me = me0+".plot_pressure_file: "
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histfile, me+"Functional only for Cartesian geometry."
	Casimir = "_CL_" in histfile

	##-------------------------------------------------------------------------
	
	## Filename parameters
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	T = filename_par(histfile, "_T") if Casimir else -S
			
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
	
	## Calculate integral pressure
	PR = -sp.integrate.cumtrapz(fx[Rind:]*Qx[Rind:], x[Rind:], initial=0.0)
	PS = -sp.integrate.cumtrapz(fx[STind:Sind]*Qx[STind:Sind], x[STind:Sind], initial=0.0); PS -= PS[-1]
	if Casimir:
		PT = -sp.integrate.cumtrapz(fx[Tind:STind]*Qx[Tind:STind], x[Tind:STind], initial=0.0)
			
	##-------------------------------------------------------------------------
	
	## Potential and WN
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U)/np.trapz(np.exp(-U),x)
	
	# PR_WN = -sp.integrate.cumtrapz(fx[Rind:]*Qx_WN[Rind:], x[Rind:], initial=0.0)
	# PS_WN = -sp.integrate.cumtrapz(fx[STind:Sind]*Qx_WN[STind:Sind], x[STind:Sind], initial=0.0); PS -= PS[-1]
	# if Casimir:
		# PT_WN = -sp.integrate.cumtrapz(fx[Tind:STind]*Qx_WN[Tind:STind], x[Tind:STind], initial=0.0)
	
	# PR_WN += 0.01*(PR_WN==0); PS_WN += 0.01*(PS_WN==0); PT_WN += 0.01*(PT_WN==0)
	# PR /= PR_WN; PS /= PS_WN; PT /= PT_WN
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	fig, axs = plt.subplots(2,1, sharex=True, figsize=(10,10))
	
	ax = axs[0]
	ax.plot(x, Qx, label=r"CN")
	ax.plot(x, Qx_WN, "r-", label="WN")
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", label=r"$U(x)$")
	
	ax.set_ylim(bottom=0.0)	
	ax.set_ylabel(r"$Q(x)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right", fontsize=fsl).get_frame().set_alpha(0.5)
	
	
	ax = axs[1]
	ax.plot(x[Rind:],      PR, label=r"$P_R$")
	ax.plot(x[STind:Sind], PS, label=r"$P_S$")
	if Casimir:
		ax.plot(x[Tind:STind], PT, label=r"$P_T$")	
	ax.plot(x, U/U.max()*ax.get_ylim()[1], "k--", label=r"$U(x)$")
	
	ax.set_ylim(bottom=0.0)	
	ax.set_xlabel(r"$x$", fontsize=fsa)
	ax.set_ylabel(r"$P(x)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="lower right", fontsize=fsl).get_frame().set_alpha(0.5)

	##-------------------------------------------------------------------------
	
	fig.tight_layout()
	fig.subplots_adjust(top=0.95)	
	title = r"Spatial PDF and Pressure. $\alpha=%.1f, R=%.1f, S=%.1f, T=%.1f$"%(a,R,S,T) if Casimir\
			else r"Spatial PDF and Pressure. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S)
	fig.suptitle(title, fontsize=fst)
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	return
	
##=============================================================================
def plot_pressure_dir(histdir, srchstr, nosave, vb):
	"""
	"""
	me = me0+".plot_pressure_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histdir, me+"Functional only for Cartesian geometry."
	Casimir = "_CL_" in histdir
	
	## File discovery
	filelist = np.sort(glob.glob(histdir+"/BHIS_CAR_*"+srchstr+"*.npy"))
	numfiles = len(filelist)
	if vb: print me+"found",numfiles,"files"

	##-------------------------------------------------------------------------
	
	A, PR, PS, PT = np.zeros([4,numfiles])
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
		
		t0 = time.time()
		
		## Assuming R, S, T are same for all files
		A[i] = filename_par(histfile, "_a")
		R = filename_par(histfile, "_R")
		S = filename_par(histfile, "_S")
		T = filename_par(histfile, "_T") if Casimir else -S
			
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
		
		## Calculate integral pressure
		PR[i] = -sp.integrate.trapz(fx[Rind:]*Qx[Rind:], x[Rind:])
		PS[i] = +sp.integrate.trapz(fx[STind:Sind]*Qx[STind:Sind], x[STind:Sind])
		PT[i] = -sp.integrate.trapz(fx[Tind:STind]*Qx[Tind:STind], x[Tind:STind])
		
		if vb: print me+"a=%.1f:\tPressure calculation %.2g seconds"%(A[i],time.time()-t0)
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	PR, PS, PT = PR[srtidx], PS[srtidx], PT[srtidx]
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	fig, ax = plt.subplots(1,1, figsize=(10,10))
	
	ax.plot(A, PR, "o-", label=r"$P_R$")
	ax.plot(A, PS, "o-", label=r"$P_S$")
	if Casimir:
		ax.plot(A, PT, "o-", label=r"$P_T$")	
	
	ax.set_xlabel(r"$\alpha$", fontsize=fsa)
	ax.set_ylabel(r"$P(\alpha)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right", fontsize=fsl).get_frame().set_alpha(0.5)
	
	ax.set_title(r"Pressure as a function of $\alpha$ for $R=%.1f,S=%.1f,T=%.1f$"%(R,S,T), fontsize=fst)
	
	if not nosave:
		plotfile = histdir+"/PA_R%.1f_S%.1f_T%.1f.jpg"%(R,S,T) if Casimir\
					else histdir+"/PA_R%.1f_S%.1f.jpg"%(R,S)
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
