me0 = "test_PlotCombinePM"

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
	Cartesian.
	Assume a MASS afile and a PRESS file both exist.
	Plot both on the same axes.
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
		plot_PM(args[0], srchstr, nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_PM(histdir, srchstr, nosave, vb):
	"""
	Plot the pressure and mass for all files in directory matching string.
	"""
	me = me0+".plot_PM: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data
		
	try:
		pressdata= np.load(histdir+"/PRESS_"+srchstr+".npz")
		massdata = np.load(histdir+"/MASS_"+srchstr+".npz")
		print me+"Pressure and mass data file found."
	except IOError:
		raise IOError, me+"No pressure or mass data found."
		
	A = massdata["A"]
	ML = massdata["ML"]/massdata["MLwn"]
	MR = massdata["MR"]/massdata["MRwn"]
	x = massdata["x"]
	Qx_WN = massdata["Qx_WN"]
	R, S, T = massdata["R"], massdata["S"], massdata["T"]
	cuspind = massdata["cuspind"]
	del massdata
	
	A = pressdata["A"]
	PR = pressdata["PR"]
	PS = pressdata["PS"]
	PT = pressdata["PT"]
	PR_WN = pressdata["PR_WN"]
	PS_WN = pressdata["PS_WN"]
	PT_WN = pressdata["PT_WN"]
	try:
		PU = pressdata["PU"]
		PU_WN = pressdata["PU_WN"]
	except KeyError:
		pass
	del pressdata
	
	##-------------------------------------------------------------------------
	
	## Add a=0 point
	if 0.0 not in A:
		nlin = np.unique(S).size
		A = np.hstack([[0.0]*nlin,A])
		PR = np.hstack([[1.0]*nlin,PR])
		PS = np.hstack([[1.0]*nlin,PS])
		PT = np.hstack([[1.0]*nlin,PT])
		try:
			PU = np.hstack([[1.0]*nlin,PU])
		except UnboundLocalError:
			pass
		
	##-------------------------------------------------------------------------
	
	## Smoothing
	sp.ndimage.gaussian_filter1d(PS[2:],sigma=0.5,order=0,output=PS[2:])
	sp.ndimage.gaussian_filter1d(PT[2:],sigma=0.5,order=0,output=PT[2:])
	sp.ndimage.gaussian_filter1d(ML[2:],sigma=0.5,order=0,output=ML[2:])
	sp.ndimage.gaussian_filter1d(MR[2:],sigma=0.5,order=0,output=MR[2:])
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	## TO PLOT ON TWINNED AXES -- deprecated
	if 0:
	
		fig, axP = plt.subplots(1,1)
		axM = axP.twinx()
	
		## Pressure normalised by WN result
		## Assuming R,S,T are the same for all files
		axP.plot(A, PS, "go-", label=r"$P_S$")
		axP.plot(A, PT, "bo-", label=r"$P_T$")

		## Mass normalised by WN result

		lL = axM.plot(A, ML, "v--", label=r"Left")
		lR = axM.plot(A, MR, "v--", label=r"Right")
	
		##----------------------------------------------------------------------------
	
		## Set axis limits
		axP.grid()
		axM.grid()
	
		## Axis labels
		axP.set_xlabel(r"$\alpha$", fontsize=fs["fsa"])
		axP.set_ylabel(r"$P/P^{\rm passive}$", fontsize=fs["fsa"])
		axM.set_ylabel(r"$M/M^{\rm passive}$", fontsize=fs["fsa"])
		
		## Inset placement
		Uleft, Ubottom, Uwidth, Uheight = [0.21, 0.69, 0.20, 0.18]
		qleft, qbottom, qwidth, qheight = [0.51, 0.20, 0.33, 0.28]
		
	##----------------------------------------------------------------------------
	## TO PLOT ON SAME AXES
	else:
	
		fig, ax = plt.subplots(1,1)
	
		## Pressure normalised by WN result
		## Assuming R,S,T are the same for all files
		ax.plot(A, PS, "go-", label=r"$P_S$")
		ax.plot(A, PT, "bo-", label=r"$P_T$")

		## Mass normalised by WN result

		lL = ax.plot(A, ML, "v--", label=r"Left")
		lR = ax.plot(A, MR, "v--", label=r"Right")
	
		##----------------------------------------------------------------------------
		
		## Number of ticks
		ntick = 10 if "_CL_" in histdir else 7
		ax.yaxis.set_major_locator(MaxNLocator(ntick))
		ax.grid()
	
		## Axis labels
		ax.set_xlabel(r"$\alpha$", fontsize=fs["fsa"])
		ax.set_ylabel(r"$P/P^{\rm passive}$ and $M/M^{\rm passive}$", fontsize=fs["fsa"])
		
		## Inset placement
		if "_CL_" in histdir:
			Uleft, Ubottom, Uwidth, Uheight = [0.57, 0.39, 0.31, 0.22]
			qleft, qbottom, qwidth, qheight = [0.20, 0.61, 0.32, 0.27]
			Rschem, Sschem, Tschem = 15.0, 2.0, 0.0
			force_x = force_clin
		elif "_ML_" in histdir:
			ax.set_ylim(bottom=0.7)
			Uleft, Ubottom, Uwidth, Uheight = [0.21, 0.14, 0.30, 0.20]
#			Uleft, Ubottom, Uwidth, Uheight = [0.57, 0.26, 0.30, 0.20]
			qleft, qbottom, qwidth, qheight = [0.19, 0.66, 0.30, 0.23]
			Rschem, Sschem, Tschem = 4.0, 2.0, 0.0
			force_x = force_mlin
	
	
	##----------------------------------------------------------------------------
	## Casimir insets
	if "_CL_" in histdir:
		## Plot potential as inset
		axin = fig.add_axes([Uleft, Ubottom, Uwidth, Uheight])
		x = np.linspace(-Rschem-2.0,+Rschem+2.0,2*x.size)
		cuspind = np.abs(x-0.5*(Sschem+Tschem)).argmin()
		fx = force_x([x,0],Rschem,Sschem,Tschem)[0]
		U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
		axin = fig.add_axes([Uleft, Ubottom, Uwidth, Uheight])
		axin.plot(x, U, "k-")
		axin.axvspan(x[0],-x[cuspind], color=lR[0].get_color(),alpha=0.2)
		axin.axvspan(-x[cuspind],x[cuspind], color=lL[0].get_color(),alpha=0.2)
		axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
		axin.set_xlim(x[0], x[-1])
		axin.set_ylim(top=3*U[cuspind])
		axin.set_yticks([1.0])
		axin.set_yticklabels(["1"])
		axin.xaxis.set_major_locator(NullLocator())
		axin.grid()
		axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
		axin.set_ylabel(r"$U/T$", fontsize = fs["fsa"]-5)
	
		## Plot q(eta) as inset
		axin = fig.add_axes([qleft, qbottom, qwidth, qheight])
		## Grab a file. Hacky. Assumes only one match.
		histfile = glob.glob(histdir+"/BHIS_CAR_CL_a5.0_*"+srchstr+"*.npy")[0]
		plot_peta_CL(histfile, fig, axin, True)
		axin.xaxis.set_major_locator(NullLocator())
		axin.yaxis.set_major_locator(NullLocator())
		
	##----------------------------------------------------------------------------
	## Single wall insets
	elif "_ML_" in histdir:
		## Plot potential as inset
		x = np.linspace(-Rschem-2.0,+Rschem+2.0,2*x.size)
		fx = force_x([x,0],R,S,T)[0]
		U = -sp.integrate.cumtrapz(fx, x, initial=0.0); U -= U.min()
		cuspind = np.abs(x-0.5*(Sschem+Tschem)).argmin()
		axin = fig.add_axes([Uleft, Ubottom, Uwidth, Uheight])
		axin.plot(x, U, "k-")
		axin.axvspan(x[0],x[cuspind], color=lL[0].get_color(),alpha=0.2)
		axin.axvspan(x[cuspind],x[-1], color=lR[0].get_color(),alpha=0.2)
		axin.set_xlim(-R-1.5, R+1.5)
		axin.set_ylim(top=3*U[cuspind])
		axin.xaxis.set_major_locator(NullLocator())
		axin.set_yticks([1.0])
		axin.set_yticklabels(["1"])
		axin.grid()
		axin.set_xlabel(r"$x$", fontsize = fs["fsa"]-5)
		axin.set_ylabel(r"$U/T$", fontsize = fs["fsa"]-5)
	
		## Plot q(eta) as inset
		axin = fig.add_axes([qleft, qbottom, qwidth, qheight])
		## Grab a file. Hacky. Assumes only one match.
		histfile = glob.glob(histdir+"/BHIS_CAR_ML_a10.0_*"+srchstr+"*.npy")[0]
		plot_peta_CL(histfile, fig, axin, True)
		axin.xaxis.set_major_locator(NullLocator())
		axin.yaxis.set_major_locator(NullLocator())
		
	##----------------------------------------------------------------------------
	##----------------------------------------------------------------------------
	
	if not nosave:
		if   "_ML_" in histfile:	geo = "ML"
		elif "_CL_" in histfile:	geo = "CL"
		
		plotfile = histdir+"/PMa_CAR_"+geo+"_R%.1f_S%.1f_T%.1f."%(R,S,T)+fs["saveext"]
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
