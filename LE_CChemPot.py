me0 = "LE_CChemPot"

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
from LE_Utils import filename_par, fs

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

mpl.rcParams['xtick.labelsize'] = fs["fsn"]
mpl.rcParams['ytick.labelsize'] = fs["fsn"]

## ============================================================================

def main():
	"""
	Calculate the mass on either side of interior wall for Casimir setup.
	Adapted from LE_CPressure.py.
	"""
	me = me0+".main: "
	t0 = time.time()
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('--str',
		dest="srchstr", default="", type="str")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	srchstr = opt.srchstr
	nosave = opt.nosave
	vb = opt.verbose
	
	## Plot directory
	if os.path.isdir(args[0]):
		plot_mass_ratio(args[0], srchstr, nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_mass_ratio(histdir, srchstr, nosave, vb):
	"""
	Read in directory of files with inner and outer regions.
	Compute mass in each region, take ratio.
	Compare with integrated and calculated white noise result. 
	"""
	me = me0+".plot_mass_ratio: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert "_CAR_" in histdir, me+"Functional only for Cartesian geometry."
	assert "_DL_" not in histdir, me+"Must have interior region."
	
	## File discovery
	filelist = np.sort(glob.glob(histdir+"/BHIS_CAR_*"+srchstr+"*.npy"))
	numfiles = len(filelist)
	assert numfiles>1, me+"Check input directory."
	if vb: print me+"found",numfiles,"files"

	##-------------------------------------------------------------------------
	
	A, ML, MR = np.zeros([3,numfiles])
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
		
		t0 = time.time()
		
		## Assume R, S, T are same for all files
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
		
		##-------------------------------------------------------------------------
		
		## Histogram
		H = np.load(histfile)
		## Spatial density
		Qx = H.sum(axis=2).sum(axis=1) / (H.sum()*(x[1]-x[0]))

		## Mass on either side of cusp: data. Left, right.
		if   "_CL_" in histfile:	cuspind = np.abs((T+S)/2-x).argmin()	## Half-domain
		elif "_ML_" in histfile:	cuspind = np.abs((T+S)/2-x).argmin()
		elif "_NL_" in histfile:	cuspind = np.abs(S-x).argmin()
		
		ML[i] = np.trapz(Qx[:cuspind],x[:cuspind])
		MR[i] = np.trapz(Qx[cuspind:],x[cuspind:])
		
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	ML = ML[srtidx]; MR = MR[srtidx]
	
	##-------------------------------------------------------------------------
	## WN result from density solution
	
	if   "_DL_" in histfile:	fx = force_dlin([x,0],R,S)[0]
	elif "_CL_" in histfile:	fx = force_clin([x,0],R,S,T)[0]
	elif "_ML_" in histfile:	fx = force_mlin([x,0],R,S,T)[0]
	elif "_NL_" in histfile:	fx = force_nlin([x,0],R,S)[0]
	else: raise IOError, me+"Force not recognised."
		
	U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
	Qx_WN = np.exp(-U) / np.trapz(np.exp(-U), x)
	
	MLwn = np.trapz(Qx_WN[:cuspind],x[:cuspind])
	MRwn = np.trapz(Qx_WN[cuspind:],x[cuspind:])
	
	##-------------------------------------------------------------------------
	"""## WN result from direct calculation	## NOT WORKING
	if   "_CL_" in histfile:	## Half-domain
		MLwnc = T+    (1-np.exp(-0.5*(0.5*(T+S))**2))
		MRwnc = R-S+1+(1-np.exp(-0.5*(0.5*(T+S))**2))
		MLwnc /= MLwnc+MRwnc; MRwnc /= MLwnc+MRwnc
	elif "_ML_" in histfile:
		MLwnc = T+R+1+(1-np.exp(-0.5*(0.5*(T+S))**2))
		MRwnc = R-S+1+(1-np.exp(-0.5*(0.5*(T+S))**2))
		MLwnc /= MLwnc+MRwnc; MRwnc /= MLwnc+MRwnc
	elif "_NL_" in histfile:	## Assume S>=0
		MLwnc = 1+(1-np.exp(-0.5*S))
		MRwnc = ( 1+(1-np.exp(-0.5*S)) ) * np.exp(S*R)
		MLwnc /= MLwnc+MRwnc; MRwnc /= MLwnc+MRwnc"""
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	fig, axs = plt.subplots(2,1, figsize=fs["figsize"])
	ms, lw = 8, 2
	
	## Mass ratio
	ax = axs[0]
	
	ax.plot(A, MR/ML, "ro-", ms=ms, lw=lw, label=r"CN")
	ax.axhline(MRwn/MLwn, c="r", ls="--", lw=lw, label=r"WN")
	
	ax.set_ylabel(r"$M_{\rm R}/M_{\rm L}$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="lower left", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	ax.set_ylim(bottom=0.0)
	
	## Mass normalised by WN result
	ax = axs[1]
	
	lL = ax.plot(A, ML/MLwn, "o-", ms=ms, lw=lw, label=r"Left")
	lR = ax.plot(A, MR/MRwn, "o-", ms=ms, lw=lw, label=r"Right")
	
	ax.set_xlabel(r"$\alpha$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$M_{\rm R,L}/M^{\rm wn}_{\rm R,L}$", fontsize=fs["fsa"])
	ax.grid()
#	ax.legend(loc="upper left", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)	

	fig.tight_layout()
	fig.subplots_adjust(top=0.90)
	
	## Plot potential as inset
	left, bottom, width, height = [0.2, 0.3, 0.2, 0.15] if "_CL_" in histfile else [0.65, 0.6, 0.2, 0.15]
	axin = fig.add_axes([left, bottom, width, height])
	axin.plot(x, U, "k-", lw=lw)
	axin.axvspan(x[0],x[cuspind], color=lL[0].get_color(),alpha=0.2)
	axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
	xlimL = -R-2.0 if "_NL_" in histfile else x[0]
	axin.set_xlim(left=xlimL, right=R+2.0)
	axin.set_ylim(top=1.0)
	axin.xaxis.set_ticklabels([])
	axin.yaxis.set_ticklabels([])
	axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
	axin.set_ylabel(r"$U$", fontsize = fs["fsa"]-5)
	
		
	title = r"Mass distribution as a function of $\alpha$ for $R=%.1g,S=%.1g,T=%.1g$"%(R,S,T) if T>=0.0\
			else r"Mass distribution as a function of $\alpha$ for $R=%.1g,S=%.1g$"%(R,S)
	fig.suptitle(title, fontsize=fs["fst"])
	
	if not nosave:
		plotfile = histdir+"/MLR_R%.1f_S%.1f_T%.1f.jpg"%(R,S,T) if T>=0.0\
					else histdir+"/MLR_R%.1f_S%.1f.jpg"%(R,S)
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()