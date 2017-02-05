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
from LE_Utils import filename_par, fs, set_mplrc

from test_etaCas import plot_peta_CL
from test_force import plot_U1D_Cartesian
from matplotlib.ticker import MaxNLocator, NullLocator

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## Plot defaults
set_mplrc(fs)

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
	parser.add_option('--noread',
		dest="noread", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	srchstr = opt.srchstr
	nosave = opt.nosave
	noread = opt.noread
	vb = opt.verbose
	
	## Plot directory
	if os.path.isdir(args[0]):
		plot_mass_ratio(args[0], srchstr, nosave, noread, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def calc_mass_ratio(histdir, srchstr, noread, vb):
	"""
	Read in directory of files with inner and outer regions.
	Compute mass in each region, take ratio.
	Compare with integrated and calculated white noise result. 
	"""
	me = me0+".calc_mass_ratio: "
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
		if   "_CL_" in histfile:	cuspind = np.abs(0.5*(T+S)-x).argmin()	## Half-domain
		elif "_ML_" in histfile:	cuspind = np.abs(0.5*(T+S)-x).argmin()
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
	## Add a=0 point
	if 0.0 not in A:
		A = np.hstack([0.0,A])
		ML = np.hstack([MLwn,ML])
		MR = np.hstack([MRwn,MR])
	
	##-------------------------------------------------------------------------
	
	### This might not be the cleanest thing to save...
	
	## SAVING
	if not noread:
		massfile = histdir+"/MASS_"+srchstr+".npz"
		np.savez(massfile, A=A, ML=ML, MR=MR, MLwn=MLwn, MRwn=MRwn, x=x, Qx_WN=Qx_WN, R=R, S=S, T=T, cuspind=cuspind)
		if vb:
			print me+"Calculations saved to",massfile
			print me+"Calculation time %.1f seconds."%(time.time()-t0)

	return {"A":A, "ML":ML, "MR":MR, "MLwn":MLwn, "MRwn":MRwn, "x":x, "Qx_WN":Qx_WN, "R":R, "S":S, "T":T, "cuspind":cuspind}
		

##=============================================================================
def plot_mass_ratio(histdir, srchstr, nosave, noread, vb):
	"""
	Plot the mass for all files in directory matching string.
	"""
	me = me0+".plot_mass_ratio: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh
		
	try:
		assert noread == False
		massdata = np.load(histdir+"/MASS_"+srchstr+".npz")
		print me+"Mass data file found:",histdir+"/MASS_"+srchstr+".npz"
	except (IOError, AssertionError):
		print me+"No mass data found. Calculating from histfiles."
		massdata = calc_mass_ratio(histdir, srchstr, noread, vb)
		
	A = massdata["A"]
	ML = massdata["ML"]
	MR = massdata["MR"]
	MLwn = massdata["MLwn"]
	MRwn = massdata["MRwn"]
	x = massdata["x"]
	Qx_WN = massdata["Qx_WN"]
	R, S, T = massdata["R"], massdata["S"], massdata["T"]
	cuspind = massdata["cuspind"]
	del massdata
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	fig, ax = plt.subplots(1,1)

	## Mass normalised by WN result

	lL = ax.plot(A, ML/MLwn, "o-", label=r"Left")
	lR = ax.plot(A, MR/MRwn, "o-", label=r"Right")

	ax.set_xlabel(r"$\alpha$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$M/M^{\rm passive}$", fontsize=fs["fsa"])
	ax.grid()
	
	##----------------------------------------------------------------------------
	## Casimir insets
	if "_CL_" in histdir:
		## Plot potential as inset
		left, bottom, width, height = [0.2, 0.6, 0.3, 0.25]
		axin = fig.add_axes([left, bottom, width, height])
	
		x = np.linspace(-x[-1],x[-1],2*x.size)
		fx = force_clin([x,0],R,S,T)[0]
		U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
		axin.plot(x, U, "k-")
		cuspind += x.size/2	## Because x.size has doubles
		axin.axvspan(x[0],-x[cuspind], color=lR[0].get_color(),alpha=0.2)
		axin.axvspan(-x[cuspind],x[cuspind], color=lL[0].get_color(),alpha=0.2)
		axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
		axin.set_xlim(-R-2, R+2)
		axin.set_ylim(top=2*U[cuspind])
		axin.xaxis.set_ticklabels([])
		axin.yaxis.set_ticklabels([])
		axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
		axin.set_ylabel(r"$U$", fontsize = fs["fsa"]-5)
	
		## Plot q(eta) as inset
		left, bottom, width, height = [0.55, 0.27, 0.33, 0.28]
		axin = fig.add_axes([left, bottom, width, height])
		## Grab a file. Hacky. Assumes only one match.
		histfile = glob.glob(histdir+"/BHIS_CAR_CL_a5.0_*"+srchstr+"*.npy")[0]
		plot_peta_CL(histfile, fig, axin, True)
		axin.xaxis.set_major_locator(NullLocator())
		axin.yaxis.set_major_locator(NullLocator())
		
	##----------------------------------------------------------------------------
	## Single wall insets
	elif "_ML_" in histdir:
		## Plot potential as inset
		fx = force_mlin([x,0],R,S,T)[0]
		U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
		left, bottom, width, height = [0.18, 0.68, 0.33, 0.20]
		axin = fig.add_axes([left, bottom, width, height])
		axin.plot(x, U, "k-")
		axin.axvspan(x[0],x[cuspind], color=lL[0].get_color(),alpha=0.2)
		axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
		axin.set_xlim(-R-1.5, R+1.5)
		axin.set_ylim(top=2*U[cuspind])
		axin.xaxis.set_major_locator(NullLocator())
		axin.yaxis.set_major_locator(NullLocator())
		axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
		axin.set_ylabel(r"$U$", fontsize = fs["fsa"]-5)
	
		## Plot q(eta) as inset
		left, bottom, width, height = [0.55, 0.35, 0.33, 0.23]
		axin = fig.add_axes([left, bottom, width, height])
		## Grab a file. Hacky. Assumes only one match.
		histfile = glob.glob(histdir+"/BHIS_CAR_ML_a10.0_*"+srchstr+"*.npy")[0]
		plot_peta_CL(histfile, fig, axin, True)
		axin.xaxis.set_major_locator(NullLocator())
		axin.yaxis.set_major_locator(NullLocator())
	
	##----------------------------------------------------------------------------
	## Double well insets
	elif "_NL_" in histdir:
		## Plot potential as inset
		fx = force_nlin([x,0],R,S)[0]
		U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
		left, bottom, width, height = [0.61, 0.30, 0.28, 0.20]
		axin = fig.add_axes([left, bottom, width, height])
		axin.plot(x, U, "k-")
		axin.axvspan(x[0],x[cuspind], color=lL[0].get_color(),alpha=0.2)
		axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
		axin.set_xlim(x[0],x[-1])
		axin.set_ylim(top=2*U[cuspind])
		axin.xaxis.set_major_locator(NullLocator())
		axin.yaxis.set_major_locator(NullLocator())
		axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
		axin.set_ylabel(r"$U$", fontsize = fs["fsa"]-5)
		
	##----------------------------------------------------------------------------
	##----------------------------------------------------------------------------
	
	if not nosave:
		if   "_ML_" in histfile:	geo = "ML"
		elif "_NL_" in histfile:	geo = "NL"
		elif "_CL_" in histfile:	geo = "CL"
		
		plotfile = histdir+"/MLR_CAR_"+geo+"_R%.1f_S%.1f_T%.1f."%(R,S,T)+fs["saveext"] if T>=0.0\
					else histdir+"/MLR_CAR_"+geo+"_R%.1f_S%.1f."%(R,S)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
