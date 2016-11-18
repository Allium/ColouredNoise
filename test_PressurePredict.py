me0 = "test_PressurePredict"

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
from LE_SPressure import calc_pressure, pdf_WN, plot_wall
from LE_SSim import force_dlin

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## ============================================================================

def main():
	"""
	Plot the bulk constant <eta^2>Q as a function of r for a single file, or...
	"""
	me = me0+".main: "
	t0 = time.time()
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('--str',
		dest="searchstr", default="", type="str")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	searchstr = opt.searchstr
	nosave = opt.nosave
	vb = opt.verbose
	
	assert "_DL_" in args[0], me+"Only dlin force supported at the moment."
		
	if os.path.isdir(args[0]):
		plot_DP_a(args[0], nosave, searchstr, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================

def get_pdf(histfile):
	"""
	Read in histfile and return normalised pdf in r, eta, psi.
	"""
	me = me0+".get_pdf: "
	
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	ebins = bins["erbins"]
	pbins = bins["epbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	eta = 0.5*(ebins[1:]+ebins[:-1])
	psi = 0.5*(pbins[1:]+pbins[:-1])
	
	## Normalise histogram and convert to density
	H = np.load(histfile)
	H /= H.sum() * (r[1]-r[0])*(eta[1]-eta[0])*(psi[1]-psi[0])
	rho = H / ( (2*np.pi)**2.0 * np.outer(r,eta)[:,:,np.newaxis] )
	Q = np.trapz(np.trapz(rho, psi, axis=2)*eta, eta, axis=1) * 2*np.pi
	
	return r, eta, psi, rho, Q

##=============================================================================

def calc_DPm(r, eta, psi, rho, Q, f, a):
	"""
	Calculate the difference in pressure according to M-(1,1).
	"""
	me = me0+".calc_DPressure: "
	
	rr = r[:,np.newaxis,np.newaxis]
	ee = eta[np.newaxis,:,np.newaxis]
	pp = psi[np.newaxis,np.newaxis,:]
	
	Q = np.trapz(np.trapz(rho, psi, axis=2)*eta, eta, axis=1) * 2*np.pi
	
	## <\eta^2\cos^2\psi>Q, <\eta^2\sin^2\psi>Q
	e2c2Q = np.trapz(np.trapz(rho * np.cos(pp)*np.cos(pp), psi, axis=2)*eta*eta * 2*np.pi*eta, eta, axis=1)
	e2s2Q = np.trapz(np.trapz(rho * np.sin(pp)*np.sin(pp), psi, axis=2)*eta*eta * 2*np.pi*eta, eta, axis=1)
	
	## \int_0^\inf (<\eta^2\cos^2\psi>-<\eta^2\sin^2\psi>-f^2)*Q/r' dr'
	intgl = sp.integrate.trapz(((e2c2Q-e2s2Q-f*f*Q)/r), r)
	
	## Pressure difference from M-(1,1)
	DPm = a*(e2c2Q[0] - f[0]*f[0]*Q[0] - intgl)
	
	return DPm, Q, e2c2Q, e2s2Q, intgl

##=============================================================================

def calc_DPi(r, Q, f, R):
	"""
	Calculate the difference in pressure according to integral.
	"""
	me = me0+"calc_DPi: "
	
	## Pressure difference from integral
	Rind = np.abs(r-R).argmin()
	DPi = -np.trapz(f[Rind:]*Q[Rind:], r[Rind:]) - np.trapz(f[:Rind]*Q[:Rind], r[:Rind])
	
	return DPi

##=============================================================================

def calc_DPp(r, Q, a, R):
	"""
	Calculate the difference in pressure according to R=S prediction, which requires fit.
	"""
	me = me0+"calc_DPp: "
	
	## Fit Q to find mean
	gauss = lambda x, m, N: N*np.exp(-0.5*(1+a)*(x-m)**2)
	fitQm = sp.optimize.curve_fit(gauss, r, Q)[0][0]
	
	## Theoretical prediction for outer pressure (see eg 161110_Update)
	func = lambda m, n: 1/(1+a)*np.exp(-0.5*(1+a)*m**2) + (m-n)*np.sqrt(np.pi/(2*(1+a)))*(1+sp.special.erf(m*np.sqrt(0.5*(1+a))))
	DPp = 1/(2*np.pi)*func(fitQm, R)/func(fitQm, 0)
	
	return DPp

##=============================================================================

def plot_DP_a(histdir, nosave, searchstr, vb):
	"""
	Compile list of DPs and plot.
	"""
	me = me0+"plot_DP_a: "
	
	filelist = np.sort(glob.glob(histdir+"/BHIS_*"+searchstr+"*.npy"))
	
	A, DPm, DPi, DPp = np.zeros((4,filelist.size))
	
	for i, histfile in enumerate(filelist):
	
		## Parameters
		A[i] = filename_par(histfile,"_a")
		R = filename_par(histfile,"_R")
		assert filename_par(histfile,"_S")==R, me+"Must have R=S for this operation."
		
		## PDF
		r, eta, psi, rho, Q = get_pdf(histfile)
		f = force_dlin(r,r,R,R)
		
		## DP
		DPm[i] = calc_DPm(r, eta, psi, rho, Q, f, A[i])[0]
		DPi[i] = calc_DPi(r, Q, f, R)
		DPp[i] = calc_DPp(r, Q, A[i], R)
	
	srtidx = A.argsort()
	A = A[srtidx]
	DPm = DPi[srtidx]
	DPi = DPi[srtidx]
	
	##-------------------------------------------------------------------------
	
	fig, ax = plt.subplots(1,1, figsize=(10,10))
	
	ax.plot(A, DPm, "o-", label=r"M-(1,1)")
	ax.plot(A, DPi, "o-", label=r"Integral")
	ax.plot(A, DPp, "o-", label=r"Predict")
	
	##-------------------------------------------------------------------------
	
	ax.set_xlabel(r"$\alpha$",fontsize=fsa)
	ax.set_ylabel(r"$P_{\rm out}-P_{\rm in}$",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best", fontsize=fsl).get_frame().set_alpha(0.5)
	ax.set_title("Pressure difference. $R=S=%.2g$"%(R),fontsize=fst)
	
	## SAVING
	plotfile = histdir+"/DPa_R"+str(R)+".jpg"
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile	
	
	return
	
	
	

##=============================================================================
if __name__ == "__main__":
	main()

