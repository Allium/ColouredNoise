me0 = "test_surfacetension"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, glob, optparse, time
import warnings

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LE_Utils import filename_par, fs, set_mplrc
from LE_SSim import force_const, force_lin, force_dcon, force_dlin,\
					force_tan, force_dtan, force_nu, force_dnu
from schem_force import plot_U3D_polar

## MPL defaults
set_mplrc(fs)

## ============================================================================

def main():
	"""
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
	parser.add_option('--jpg',
		dest="savejpg", default=False, action="store_true")
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
	
	if opt.savejpg: fs["saveext"]="jpg"
	

	if os.path.isdir(args[0]):
		plot_energy_dir(args[0], srchstr, logplot, nosave, noread, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

	
##=============================================================================
def calc_energy_dir(histdir, srchstr, noread, vb):
	"""
	Calculate the steady state energy for all files in directory matching string.
	The 
	"""
	me = me0+".calc_energy_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	
	## Dir pars
	assert ("_POL_" in histdir or "_CIR_" in histdir), me+"Functional only for polar geometry."
	
	## File discovery
	filelist = np.sort(glob.glob(histdir+"/BHIS_*"+srchstr+"*.npy"))
	numfiles = len(filelist)
	assert numfiles>1, me+"Check input directory."
	if vb: print me+"found",numfiles,"files"

	##-------------------------------------------------------------------------
	
	A, R, S, E, E_WN = np.zeros([5,numfiles])
	
	## Retrieve data
	for i, histfile in enumerate(filelist):
		
		ti = time.time()
		
		## Assuming R, S, T are same for all files
		A[i] = filename_par(histfile, "_a")
		R[i] = filename_par(histfile, "_R")
		S[i] = filename_par(histfile, "_S")
			
		## Space
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		rbins = bins["rbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		erbins = bins["erbins"]
		er = 0.5*(erbins[1:]+erbins[:-1])
		
		## Wall indices
		Rind, Sind = np.abs(r-R[i]).argmin(), np.abs(r-S[i]).argmin()
			
		##-------------------------------------------------------------------------
		
		## Load histogram, normalise
		H = np.load(histfile)
		try: H = H.sum(axis=2)
		except ValueError: pass
		H = np.trapz(H, er, axis=1)
		## Noise dimension irrelevant here; convert to *pdf*
		rho = H/(2*np.pi*r) / np.trapz(H, x=r, axis=0)
		
		##-------------------------------------------------------------------------
		
		## Choose force
		f = force_dlin(r,r,R[i],S[i])
		U = -sp.integrate.cumtrapz(f, r, initial=0.0); U -= U.min()
		
		## Calculate energy
		E[i] = sp.integrate.trapz(U*rho*2*np.pi*r, r)
		
		if vb: print me+"a=%.1f:\tEnergy calculation %.2g seconds"%(A[i],time.time()-ti)
		
		## Potential
		rho_WN = np.exp(-U) / np.trapz(np.exp(-U), r)
		## WN energy
		E_WN[i] = sp.integrate.trapz(U*rho_WN, r)
		
	##-------------------------------------------------------------------------
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	R, S = R[srtidx], S[srtidx]
	E, E_WN = E[srtidx], E_WN[srtidx]
		
	##-------------------------------------------------------------------------
		
	## SAVING
	if not noread:
		pressfile = histdir+"/E_"+srchstr+".npz"
		np.savez(pressfile, A=A, R=R, S=S, E=E, E_WN=E_WN)
		if vb:
			print me+"Calculations saved to",pressfile
			print me+"Calculation time %.1f seconds."%(time.time()-t0)

	return {"A":A,"R":R,"S":S,"E":E,"E_WN":E_WN}
		

##=============================================================================
def plot_energy_dir(histdir, srchstr, logplot, nosave, noread, vb):
	"""
	Plot the energy for all files in directory matching string.
	"""
	me = me0+".plot_pressure_dir: "
	t0 = time.time()
	
	##-------------------------------------------------------------------------
	## Read in existing data or calculate afresh
		
	try:
		assert noread == False
		pressdata = np.load(histdir+"/E_"+srchstr+".npz")
		print me+"Energy data file found:",histdir+"/E_"+srchstr+".npz"
	except (IOError, AssertionError):
		print me+"No energy data found. Calculating from histfiles."
		pressdata = calc_energy_dir(histdir, srchstr, noread, vb)
		
	A = pressdata["A"]
	R = pressdata["R"]
	S = pressdata["S"]
	E = pressdata["E"]
	E_WN = pressdata["E_WN"]
	del pressdata
	
	##-------------------------------------------------------------------------
	
	## PLOTTING
	
	t0 = time.time()
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	## Add a=0 point
#	if 0.0 not in A:
#		nlin = np.unique(S).size
#		A = np.hstack([[0.0]*nlin,A])
#		R = np.hstack([R[:nlin],R])
#		E = np.hstack([[1.0]*nlin,E])
#		E_WN = np.hstack([[1.0]*nlin,E_WN])

	## Eliminate R=0
	if logplot:
		idx = (R!=0.0)
		A = A[idx]
		R = R[idx]
		E = E[idx]
		E_WN = E_WN[idx]
			
	##-------------------------------------------------------------------------
	
	## Assume S=R for now
	
	plotfile = histdir+"/E_R=S."+fs["saveext"]
	
#	for Ri in np.unique(R):
#		idx = (R==Ri)	## Pick out each alpha
#		Aj, Ej, E_WNj = A[idx], E[idx], E_WN[idx]
#		idx = Aj.argsort()	## Sort according to A
#		ax.plot(Aj[idx], Ej[idx]/E_WNj[idx], "o-", label=r"R=%.1f"%(Ri))
	for Ai in np.unique(A):
		idx = (A==Ai)	## Pick out each alpha
		Rj, Ej, E_WNj = R[idx], E[idx], E_WN[idx]
		idx = Rj.argsort()	## Sort according to R
		ax.plot(Rj[idx], Ej[idx]/E_WNj[idx], "o-", label=r"\alpha=%.1f"%(Ai))
		
	RR = np.linspace(1,R.max(),201)
#	ax.plot(RR,1/(RR),"k:",lw=3,label=r"$R^{-1}$")
					
	##-------------------------------------------------------------------------
	
	## Plot appearance
			
	if logplot:
		ax.set_xscale("log")
		ax.set_yscale("log")
#		ax.set_ylim(top=1e1)
		plotfile = plotfile[:-4]+"_loglog."+fs["saveext"]
	
	ax.set_xlabel(r"$R$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$E(\alpha,R)/E^{\rm passive}$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="best", ncol=2, fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Plotting time %.1f seconds."%(time.time()-t0)
	
	return
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
