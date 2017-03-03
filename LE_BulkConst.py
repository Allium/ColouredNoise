me0 = "LE_SBulkConst"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

import matplotlib as mpl
if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	mpl.use("Agg")
from matplotlib import pyplot as plt

from LE_Utils import filename_pars, filename_par, fs, set_mplrc
from LE_SPressure import calc_pressure, pdf_WN, plot_wall
from LE_SSim import force_dlin

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

set_mplrc(fs)

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
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('--str',
		dest="searchstr", default="", type="str")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	plotall = opt.plotall
	searchstr = opt.searchstr
	nosave = opt.nosave
	verbose = opt.verbose
		
	if os.path.isfile(args[0]):
		plot_file(args[0], nosave, verbose)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		assert len(filelist)>1, me+"Check directory."
		if verbose: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_file(histfile, nosave, verbose)
			plt.close()
	elif os.path.isdir(args[0]):
		plot_dir(args[0], nosave, searchstr, verbose)
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile, nosave, vb):
	"""
	"""
	me = me0+".plot_file: "

	## Get pars from filename
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	
	## Calculate quantities
	r, Q, BCout, BCin, Pout, Pin, e2c2Q, e2s2Q, intgl = bulk_const(histfile)[:9]
	
	## Wall indices	
	Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()
	
	##-------------------------------------------------------------------------
	
	## PLOT
	fig = plt.figure(figsize=fs["figsize"]); ax = fig.gca()
		
	## Data
	ax.plot(r,Q, label=r"$Q(r)$")
	ax.plot(r,BCout, label=r"BC (out)", lw=2)
	if S>0.0:
		ax.plot(r,BCin,label=r"BC (in)", lw=2)
	ax.plot(r,e2c2Q, label=r"$\langle\eta^2\cos^2\psi\rangle Q(r)$")
	ax.plot(r,e2s2Q, label=r"$\langle\eta^2\sin^2\psi\rangle Q(r)$")
	ax.plot(r,intgl, label=r"$\int_0^r\frac{1}{r^\prime}(\cdots)Q\,dr^\prime$")
	
	## Potential
	ymax = ax.get_ylim()[1]
	if "_DL" in histfile:
		U = np.hstack([np.linspace(S-r[0],0,Sind)**2,np.zeros(Rind-Sind),np.linspace(0.0,r[-1]-R,r.size-Rind)**2])
		ax.plot(r, U/U.max()*ymax, "k--", label=r"$U(r)$")
	
	## Indicate bulk region
	ax.axvspan(S,R, color="yellow",alpha=0.2)
	ax.axvline(S, c="k",lw=2);	ax.axvline(R, c="k",lw=2)
	
	##-------------------------------------------------------------------------
	
	## ATTRIBUTES
	
	ax.set_xlim(left=r[0],right=r[-1])

	ax.set_xlabel("$r$",fontsize=fs["fsa"])
	ax.set_ylabel("Rescaled variable",fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="lower right",fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	fig.suptitle(r"Bulk Constant. $\alpha=%.1f, R=%.1f, S=%.1f$."%(a,R,S),fontsize=fs["fst"])
	
	## SAVE
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+".jpg"
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
	
	##-------------------------------------------------------------------------
	
	return plotfile
	
##=============================================================================
def plot_dir(histdir, nosave, searchstr, vb):
	"""
	For each file in directory, calculate the pressure in both ways for both walls
	(where applicable) and plot against alpha.
	"""
	me = me0+".plot_dir: "
	
	ftype = filename_pars(histdir)["ftype"]
	
	filelist = np.sort(glob.glob(histdir+"/BHIS_*"+searchstr+"*.npy"))
	numfiles = filelist.size
	if vb: print me+"Found",numfiles,"files."
	
	## Initialise arrays
	A,Cout,Cin,Pout,Pin = np.zeros([5,numfiles])	
	
	## Retrieve data
	for i,histfile in enumerate(filelist):
		
		t0 = time.time()
		r, Q, BCout, BCin, Pout[i], Pin[i], e2c2Q, e2s2Q, intgl, pars = bulk_const(histfile)
		if vb: print me+"a=%.1f:\tBC calculation %.2g seconds"%(pars["a"],time.time()-t0)
		A[i], R, S = pars["a"], pars["R"], pars["S"]	
		fpars = [R,S,pars["lam"],pars["nu"]]
		Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()
		Cout[i] = BCout[Sind+10:Rind-10].mean() if Rind!=Sind else BCout[max(0,Sind):Rind+1].mean()
		if S>0.0:
			Cin[i] = BCin[Sind+10:Rind-10].mean() if Rind!=Sind else BCin[max(0,Sind):Rind+1].mean()
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]
	Pout, Pin = Pout[srtidx], Pin[srtidx]
	Cout, Cin = Cout[srtidx], Cin[srtidx]
	
	##-------------------------------------------------------------------------
	
	## Calculate white noise pressure and pdf
	## Assume all files have same R & S. P_WN independent of alpha.
	r_WN = np.linspace(r[0],r[-1],2*r.size+1)
	Rind_WN, Sind_WN = np.abs(r_WN-R).argmin(), np.abs(r_WN-S).argmin()			
	p_WN = calc_pressure(r_WN,pdf_WN(r_WN,fpars,ftype),ftype,fpars,True)
	Pout_WN = p_WN[-1] - p_WN.min()	## For outer wall
	Pin_WN  = p_WN[0]  - p_WN.min()	## For inner wall
	
	## NORMALISE
	Pout /= Pout_WN
	Cout /= Pout_WN
	if S>0.0:
		Pin /= Pin_WN
		Cin /= Pin_WN
	
	##-------------------------------------------------------------------------
	
	## FIT -- A*C -- fit in log coordinates
	fitfunc = lambda x, B, nu: B + nu*x
	
	## Outer C and P
	fitCout = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(A*Cout), p0=[+1.0,-1.0])[0]
	fitPout = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(Pout),   p0=[+1.0,-1.0])[0]
	if vb:	print me+"nu_Cout = ",fitCout.round(3)[1],"\t nu_Pout = ",fitPout.round(3)[1]
	
	## Inner C and P
	if S>0.0:
		fitCin = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(A*Cin), p0=[+1.0,-1.0])[0]
		fitPin = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(Pin),  p0=[+1.0,-1.0])[0]
		if vb:	print me+"nu_Cin = ",fitCin.round(3)[1],"\t nu_Pin = ",fitPin.round(3)[1]
	
	##-------------------------------------------------------------------------
	
	## Add A=0 point
	if 1:
		A = np.hstack([[0.0],A])
		Pout = np.hstack([[1.0],Pout])
		Pin = np.hstack([[1.0],Pin])
		Cout = np.hstack([[1.0],A[1:]*Cout])
		Cin = np.hstack([[1.0],A[1:]*Cin])
		## I HAVE REMOVED A* FROM PLOT LINES
		
	##-------------------------------------------------------------------------
	
	## PLOT DATA
	fig = plt.figure(figsize=fs["figsize"]); ax = fig.gca()
	
	linePo = ax.plot(1+A, Pout, "o--", label=r"$-\int_{\rm bulk}^{\infty} fQ\,{\rm d}r$")
	lineCo = ax.plot(1+A, Cout, "v--", label=r"Eq. (20) (outer)")
	if S>0.0:
		linePi = ax.plot(1+A, Pin, "o--", label=r"$-\int_{0}^{\rm bulk} fQ\,{\rm d}r$")
		lineCi = ax.plot(1+A, Cin, "v--", label=r"Eq. (20) (inner)")
		
	## PLOT FIT
#	## Cout, Pout
#	ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitPout)), linePo[0].get_color()+"--", lw=1,
#			label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitPout[0]),fitPout[1]))
#	ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitCout)), lineCo[0].get_color()+"--", lw=1,
#			label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitCout[0]),fitCout[1]))
#	## Cin, Pin
#	if S>0.0:
#		ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitPin)), linePi[0].get_color()+"--", lw=1,
#				label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitPin[0]),fitPin[1]))
#		ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitCin)), lineCi[0].get_color()+"--", lw=1,
#				label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitCin[0]),fitCin[1]))
		
	
#	## PLOT PREDICTION for R=S
#	if R == S:
#		## Outer: see notes 09/11/2016
#		b = lambda c, m: 1/(c*(A+1))*np.exp(-0.5*c*(A+1)*m*m) +\
#							+ m*np.sqrt(np.pi/(2*c*(A+1)))*(1+sp.special.erf(m*np.sqrt(0.5*c*(A+1))))
#		Pout = 1/(2*np.pi*np.pi)*b(0.01, 0.1)/b(0.01, R+0.1)
#		ax.plot(1+A, Pout/Pout_WN, "m:", label = r"Predicted $P_{\rm out}$", lw=2)
	
	##-------------------------------------------------------------------------
	
	## ACCOUTREMENTS
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlim(1,1+A[-1])
	ax.set_ylim(top=1e1)
	
	ax.set_xlabel(r"$1+\alpha$",fontsize=fs["fsa"])
	ax.set_ylabel(r"$P(\alpha)/P^{\rm passive}$",fontsize=fs["fsa"])
	ax.grid()
#	ax.legend(loc="best", fontsize=(fs["fsl"]/2 if S>0.0 else fs["fsl"])).get_frame().set_alpha(0.5)
	ax.legend(loc="best", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
#	ax.set_title("Pressure normalised by WN result. $R=%.1f, S=%.1f.$"%(fpars[0],fpars[1]),fontsize=fs["fst"])
	
	## SAVING
	plotfile = histdir+"/QEe2_Pa_R"+str(fpars[0])+"_S"+str(fpars[1])+"."+fs["saveext"]
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

	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	pars = filename_pars(histfile)
	ftype, lam, nu = pars["ftype"], pars["lam"], pars["nu"]
	
	psifile = "_psi" in histfile
	phifile = "_phi" in histfile
	
	H = np.load(histfile)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	
	## Space and load histogram
	rbins = bins["rbins"]
	ebins = bins["erbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	eta = 0.5*(ebins[1:]+ebins[:-1])
	if psifile:
		pbins = bins["epbins"]
		psi = 0.5*(pbins[1:]+pbins[:-1])
	## For old _phi files
	if phifile:
		epbins = bins["epbins"]
		H = H.sum(axis=2) * (epbins[1]-epbins[0])
	
	## Spatial arrays with dimensions commensurate to rho
	if psifile:
		rr = r[:,np.newaxis,np.newaxis]
		ee = eta[np.newaxis,:,np.newaxis]
		pp = psi[np.newaxis,np.newaxis,:]
		dV = (r[1]-r[0])*(eta[1]-eta[0])*(psi[1]-psi[0])	## Assumes regular grid
	else:
		rr = r[:,np.newaxis]
		ee = eta[np.newaxis,:]
		dV = (r[1]-r[0])*(eta[1]-eta[0])	## Assumes regular grid

	## Wall indices
	Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()

	## --------------------------------------------------------------------
	
	## Normalise histogram and convert to density
	H /= H.sum()*dV
	rho = H / ( (2*np.pi)**2.0 * rr*ee )
	
	## Marginalise over eta and calculate BC
	if psifile:
		## Radial density
		Q = np.trapz(np.trapz(rho, psi, axis=2)*eta, eta, axis=1) * 2*np.pi
		assert "_DL_" in histfile, me+"Only dlin force supported at the moment."
		f = force_dlin(r,r,R,S)
		## <\eta^2\cos^2\psi>Q, <\eta^2\sin^2\psi>Q
		e2c2Q = np.trapz(np.trapz(rho * np.cos(pp)*np.cos(pp), psi, axis=2)*eta*eta * 2*np.pi*eta, eta, axis=1)
		e2s2Q = np.trapz(np.trapz(rho * np.sin(pp)*np.sin(pp), psi, axis=2)*eta*eta * 2*np.pi*eta, eta, axis=1)
		## \int_0^r (<\eta^2\cos^2\psi>-<\eta^2\sin^2\psi>-f^2)*Q/r' dr'
		intgl = sp.integrate.cumtrapz(((e2c2Q-e2s2Q-f*f*Q)/r), r, axis=0, initial=0.0)
		
		## Line sometimes gets choppy towards r=0, especially for low alpha and S.
		## This throws the evaluation of e2c2Q at r=0, necessary for BCin.
		## Here, I fit a low-r portion of e2c2Q to a quadratic and use that intercept.
		if (S<=2.0 and S>0.0 and a<2.0):
			fitfunc = lambda x, c2, c1, c0: c2*x*x + c1*x + c0
			fitE2C2Q = sp.optimize.curve_fit(fitfunc, r[10:30], e2c2Q[10:30])[0]
			e2c2Q[0] = fitfunc(r[0], *fitE2C2Q)
			
		## Bulk constants
		BCout = e2c2Q + intgl - intgl[-1]		## Attention to integral limits
		BCin  = e2c2Q + intgl - (e2c2Q[0]-f[0]*f[0]*Q[0]) 
		
	else:
		## Radial density
		Q = np.trapz(H,eta,axis=1) / (2*np.pi*r)
		## Bulk constant <eta^2> Q
		BC = np.trapz(rho * eta*eta * 2*np.pi*eta, eta, axis=1)
	
	## --------------------------------------------------------------------

	## psi diagnostics plot
	if (0 and psifile):
		plot_psi_diagnostics(rho,Q,r,eta,psi,rr,ee,pp,R,Rind,S,Sind,a,histfile,showfig=False)			
					
	## Integral pressure calculation
	Pout = +calc_pressure(r[Rind:],Q[Rind:],ftype,[R,S,lam,nu])	## For outer wall
	Pin  = -calc_pressure(r[:Sind],Q[:Sind],ftype,[R,S,lam,nu])	## For inner wall
				
	return r, Q, BCout, BCin, Pout, Pin, e2c2Q, e2s2Q, intgl, pars
	
##=============================================================================
def plot_psi_diagnostics(rho,Q,r,eta,psi,rr,ee,pp,R,Rind,S,Sind,a,histfile,showfig=False):
	"""
	Plot things to do with the psi angle
	"""
	me = me0+".plot_psi_diagnostics: "
	fig, axs = plt.subplots(3,1, figsize=(9,12))
	## --------------------------------------------------------------------
	## PDF
	ax = axs[0]
	pdfp = np.trapz(np.trapz(rho *ee*rr, eta, axis=1), r, axis=0) *2*np.pi
	lpdfp = ax.plot(psi, pdfp, label=r"$p(\psi)$")
	lpdfp = ax.plot(psi, pdfp[::-1], label=r"$p(-\psi)$")
	ax.vlines([-2*np.pi,-np.pi,0,+np.pi,+2*np.pi], *ax.get_ylim())
	ax.set_xlabel(r"$\psi$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$p(\psi)$", fontsize=fs["fsa"])
	ax.grid(); ax.legend(loc="upper left", fontsize=fs["fsl"])
	## --------------------------------------------------------------------
	## Angles	
	ax = axs[1]
	Ecp = np.trapz(np.trapz(rho * np.cos(pp), psi, axis=2)*eta, eta, axis=1)*2*np.pi / (Q+(Q==0))
	Esp = np.trapz(np.trapz(rho * np.sin(pp), psi, axis=2)*eta, eta, axis=1)*2*np.pi / (Q+(Q==0))
	lEcp = ax.plot(r, Ecp, label=r"$\langle\cos\psi\rangle(r)$")
	lEsp = ax.plot(r, Esp, label=r"$\langle\sin\psi\rangle(r)$")
	ax.set_ylim(top=np.ceil(Ecp.max()))
	ax.set_xlabel(r"$r$", fontsize=fs["fsa"])
	ax.set_ylabel(r"Pointing", fontsize=fs["fsa"])
	ax.grid(); ax.legend(loc="upper left", fontsize=fs["fsl"])
	## --------------------------------------------------------------------
	## Force (assume dlin)	
	ax = axs[2]
	Eecp = np.trapz(np.trapz(rho * ee*np.cos(pp), psi, axis=2)*eta, eta, axis=1)*2*np.pi / (Q+(Q==0))
	Eesp = np.trapz(np.trapz(rho * ee*np.sin(pp), psi, axis=2)*eta, eta, axis=1)*2*np.pi / (Q+(Q==0))
	lEecp = ax.plot(r, Eecp, label=r"$\langle\eta\,\cos\psi\rangle(r)$")
	lEesp = ax.plot(r, Eesp, label=r"$\langle\eta\,\sin\psi\rangle(r)$")
	f = -np.hstack([np.linspace(r[0]-S,0.0,Sind),np.zeros(Rind-Sind),np.linspace(0,r[-1]-R,r.size-Rind)])	## Assume dlin
	ax.plot(r, -f, "k--", label=r"$-f(r)$")
	ax.set_ylim(top=np.ceil(Eecp.max()))
	ax.set_xlabel(r"$r$", fontsize=fs["fsa"])
	ax.set_ylabel(r"Force", fontsize=fs["fsa"])
	ax.grid(); ax.legend(loc="upper left", fontsize=fs["fsl"])
	## --------------------------------------------------------------------
	fig.suptitle(r"$\psi$ diagnostics. $\alpha=%.1f$, $R=%.1f$, $S=%.1f$"%(a,R,S), fontsize=fs["fst"])
	fig.tight_layout()
	fig.subplots_adjust(top=0.93)
	plotfile = os.path.dirname(histfile)+"/PSI_a%.1f_R%.1f_S%.1f.jpg"%(a,R,S)
	fig.savefig(plotfile)
	print me+"psi diagnostics plot saved to "+plotfile
	if showfig:	plt.show()
	## --------------------------------------------------------------------
	return

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
