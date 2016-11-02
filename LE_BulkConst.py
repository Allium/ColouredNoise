me0 = "LE_BulkConst"

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
from LE_Pressure import pressure_x
from LE_SPressure import calc_pressure, pdf_WN, plot_wall

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

	## CALCULATIONS
	x, Q, e2E, BC, p, pars = bulk_const(histfile)
	ftype = pars["ftype"]
	fpars = [pars["X"]] if pars["geo"] is "1D" else [pars["R"],pars["S"],pars["lam"],pars["nu"]]
	Rind = np.abs(x-pars["R"]).argmin()
	ord = "r" if pars["geo"] == "CIR" else "x"
	
	## PLOT
	fig = plt.figure(); ax = fig.gca()
	plot_wall(ax, ftype, fpars, x)
	ax.plot(x,Q/Q.mean(),label=r"$Q("+ord+")$")
	ax.plot(x,e2E/e2E.mean(),label=r"$\langle\eta^2\cos^2\psi\rangle("+ord+")$")
	ax.plot(x,BC/BC.mean(),label=r"$Q\cdot\langle\eta^2\cos^2\psi\rangle$")
	
	ax.axvspan(pars["S"],pars["R"],color="yellow",alpha=0.2)
	
	## ATTRIBUTES
	ax.set_xlim(left=0.0,right=x[-1])
#	if ftype[0]!="d":
#		if innerwall:	ymax = 3.0*np.median((BC/BC.mean())[np.abs(fpars[0]-x).argmin():])
#		else:			ymax = 3.0*np.median((BC/BC.mean())[:np.abs(fpars[0]-x).argmin()+1])
#	else:	ymax = 3.0*np.median((BC/BC.mean())[np.abs(fpars[1]-x).argmin():np.abs(fpars[0]-x).argmin()+1])
#	ymin = float("%.1g"%(BC.min()/BC.mean()))	## Choose max of Q and BC after wall
#	ymax = float("%.1g"%(max(Q[Rind:].max()/Q.mean(),BC[Rind:].max()/BC.mean())))	## Choose max of Q and BC after wall
#	ax.set_ylim(bottom=ymin,top=ymax)
	ax.set_xlabel("$"+ord+"$",fontsize=fsa)
	ax.set_ylabel("Rescaled variable",fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper left",fontsize=fsl+2)
	fig.suptitle(r"Bulk Constant. $\alpha = "+str(pars["a"])+"$.",fontsize=fst)
	
	## SAVE
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+".jpg"
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
	
	return plotfile
	
##=============================================================================
def plot_dir(histdir, nosave, searchstr, vb):
	"""
	"""
	me = me0+".plot_dir: "
	
	dirpars = filename_pars(histdir)
	geo = dirpars["geo"]
	ftype = dirpars["ftype"]
	
	filelist = np.sort(glob.glob(histdir+"/BHIS_*"+searchstr+"*.npy"))
	numfiles = len(filelist)
	if vb: print me+"Found",numfiles,"files."
	
	## Initialise arrays
	A,X,C,P,P_WN = np.zeros([5,numfiles])	
	
	## Retrieve data
	for i,histfile in enumerate(filelist):
		
		t0 = time.time()
		[x, Hx, e2E, BC, p, pars] = bulk_const(histfile)
		if vb: print me+"File %i of %i: BC calculation %.2g seconds"%(i,numfiles,time.time()-t0)
		A[i] = pars["a"]
		P[i] = p
		
		
		"""if geo == "1D":
			X[i] = pars["X"]
			Xidx = np.argmin(np.abs(x-X[i]))
			force = 0.5*(np.sign(X[i]-x)-1)* ((x-X[i]) if ftype is "lin" else 1)
			P_WN[i] = -(force*pdf_WN(x,R[i],ftype)).sum()*(x[1]-x[0])
			C[i] = BC[:widx].mean()"""
			
		if geo == "CIR":
			fpars = [pars["R"],pars["S"],pars["lam"],pars["nu"]]
			Rind, Sind = np.abs(x-fpars[0]).argmin(), np.abs(x-fpars[1]).argmin()
			C[i] = BC[Sind+10:Rind-10].mean() if Rind!=Sind else BC[max(0,Sind-5):Rind+5].mean()  ##MESS
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]; P = P[srtidx]; P_WN = P_WN[srtidx]; C = C[srtidx]
	
	## Calculate white noise pressure and pdf
	## Assume all files have same R & S. P_WN independent of alpha.
	r_WN = np.linspace(x[0],x[-1],2*x.size+1)
	Rind_WN, Sind_WN = np.abs(r_WN-fpars[0]).argmin(), np.abs(r_WN-fpars[1]).argmin()			
	p_WN = calc_pressure(r_WN,pdf_WN(r_WN,fpars,ftype),ftype,fpars,True)
	P_WN = p_WN[-1] - p_WN.min()	## For outer wall
#	P_WN = p_WN[0]  - p_WN.min()	## For inner wall
	
	## NORMALISE
	P /= P_WN
	C /= P_WN
	
	## FIT -- A*C -- fit in log coordinates
	fitfunc = lambda x, B, nu: B + nu*x
	fitBC = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(A*C), p0=[+1.0,-1.0])[0]
	## FIT -- P_int
	fitP = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(P), p0=[+1.0,-1.0])[0]
	if vb:	print me+": nu_BC = ",fitBC.round(3)[1],"\t nu_Int = ",fitP.round(3)[1]
	
	## PLOT DATA AND FIT
	fig = plt.figure(); ax = fig.gca()
	
	ax.plot(1+A, P, "o-", label=r"$-\int_{\rm bulk}^{\infty} fQ\,{\rm d}r$")
	ax.plot(1+A, A*C, "o-", label=r"$\alpha Q\langle\eta^2\cos^2\psi\rangle|_{\rm bulk}$")

	ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitP)), "b--", lw=1,
			label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitP[0]),fitP[1]))
	ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitBC)), "g--", lw=1,
			label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitBC[0]),fitBC[1]))
	
	## Prediction for R=S
	if fpars[0] == fpars[1]:
		R = fpars[0]
		Pout = 1/(4*np.pi*(1+A)*(1/(1+A)*np.exp(-0.5*(1+A)*R*R)+\
							+np.sqrt(np.pi/(2*(1+A)))*R*(sp.special.erf(np.sqrt((0.5*(1+A))*R)+1))))
		ax.plot(1+A, Pout/P_WN, ":", label = r"Predicted $P_{\rm out}$")
		if R <= 2*np.sqrt(np.log(10)):
			ax.plot(1+A, Pout/P_WN * (1-np.exp(-0.5*(1+A)*R*R)), ":", label = r"Predicted $P_{\rm in}$")
	
	## ACCOUTREMENTS
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlim(1,1+A[-1])
	ax.set_ylim(1e-2,1e1)
	
	ax.set_xlabel(r"$1+\alpha$",fontsize=fsa)
	ax.set_ylabel(r"$P$",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best")
	fig.suptitle("Pressure normalised by WN result. $R=%.2g, S=%.2g.$"%(fpars[0],fpars[1]),fontsize=fst)
	
	## SAVING
	plotfile = histdir+"/QEe2_Pa_R"+str(fpars[0])+"_S"+str(fpars[1])+".jpg"
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
		
	return plotfile
	
##=============================================================================
##=============================================================================

def bulk_const(histfile):
	"""
	"""
	me = me0+",bulk_const: "

	try:
		pars = filename_pars(histfile)
		[a,X,R,S,D,lam,nu,ftype,geo] = [pars[key] for key in ["a","X","R","S","D","lam","nu","ftype","geo"]]
	except:	## Disc wall surrounded by bulk
		a = filename_par(histfile, "_a")
		S = filename_par(histfile, "_S")
		geo = "INCIR"; ftype = "linin"
		R,lam,nu = 100,None,None
		pars = {"a":a,"R":R,"S":S,"lam":lam,"nu":nu,"ftype":ftype,"geo":geo}
	
	psifile = "_psi" in histfile
	phifile = "_phi" in histfile
	
	H = np.load(histfile)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	
	"""## 1D sim
	if geo == "1D":
		xbins = bins["xbins"]
		ebins = bins["ebins"]
		x = 0.5*(xbins[1:]+xbins[:-1])
		eta = 0.5*(ebins[1:]+ebins[:-1])
		H /= np.trapz(np.trapz(H,x=x,axis=1),x=eta,axis=0)
		## Integrate over eta
		Q = np.trapz(H,x=eta,axis=0)
		## Force
		if ftype == "const":	force = force_1D_const(x,X,D)
		elif ftype == "lin":	force = force_1D_lin(x,X,D)
		p = pressure_x(force,Q,x)
		e2E = np.trapz(((H/Q).T*(eta*eta)).T,x=eta,axis=0)
		BC = Q*e2E"""
		
	## Circular sim
	if "CIR" in geo:
		
		## Space and load histogram
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		etar = 0.5*(erbins[1:]+erbins[:-1])
		if psifile:
			epbins = bins["epbins"]
			etap = 0.5*(epbins[1:]+epbins[:-1])
		## For old _phi files
		if phifile:
			epbins = bins["epbins"]
			H = H.sum(axis=2) * (epbins[1]-epbins[0])
		
		## Spatial arrays with dimensions commensurate to rho
		if psifile:
			rr = r[:,np.newaxis,np.newaxis]
			ee = etar[np.newaxis,:,np.newaxis]
			pp = etap[np.newaxis,np.newaxis,:]
			dV = (r[1]-r[0])*(etar[1]-etar[0])*(etap[1]-etap[0])	## Assumes regular grid
		else:
			rr = r[:,np.newaxis]
			ee = etar[np.newaxis,:]
			dV = (r[1]-r[0])*(etar[1]-etar[0])	## Assumes regular grid
	
		## --------------------------------------------------------------------
		
		## Normalise histogram and convert to density
		H /= H.sum()*dV
		rho = H / ( (2*np.pi)**2.0 * rr*ee )
		
		## Marginalise over eta and calculate BC
		if psifile:
			## Radial density
			Q = np.trapz(np.trapz(rho, etap, axis=2)*etar, etar, axis=1) * 2*np.pi
			## Bulk constant <eta^2 cos^2psi> Q
			BC = np.trapz(np.trapz(rho * np.cos(pp)*np.cos(pp), etap, axis=2)*etar*etar*etar, etar, axis=1)
		else:
			## Radial density
			Q = np.trapz(H,etar,axis=1) / (2*np.pi*r)
			## Bulk constant <eta^2> Q
			BC = np.trapz(rho * etar*etar * 2*np.pi*etar, etar, axis=1)
		
		## Calculate average eta
		e2E = BC / (Q+(Q==0))	  ## Avoid /0 warning (numerator is 0 anyway)
		
		## --------------------------------------------------------------------

		## Wall indices
		Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()

		## psi diagnostics plot
		if (0 and psifile):
			plot_psi_diagnostics(rho,Q,r,etar,etap,rr,ee,pp,R,Rind,S,Sind,a,histfile,showfig=False)			
						
		## Integral pressure calculation
		## Integrate from bulk to infinity (outer wall)
		p = +calc_pressure(r[Rind:],Q[Rind:],ftype,[R,S,lam,nu])	## For outer wall
		# p = -calc_pressure(r[:Sind],Q[:Sind],ftype,[R,S,lam,nu])	## For inner wall
				
	return [r, Q, e2E, BC, p, pars]
	
##=============================================================================
def plot_psi_diagnostics(rho,Q,r,etar,etap,rr,ee,pp,R,Rind,S,Sind,a,histfile,showfig=False):
	"""
	Plot things to do with the psi angle
	"""
	me = me0+".plot_psi_diagnostics: "
	fig, axs = plt.subplots(3,1, figsize=(9,12))
	## --------------------------------------------------------------------
	## PDF
	ax = axs[0]
	pdfp = np.trapz(np.trapz(rho *ee*rr, etar, axis=1), r, axis=0) *2*np.pi
	lpdfp = ax.plot(etap, pdfp, label=r"$p(\psi)$")
	lpdfp = ax.plot(etap, pdfp[::-1], label=r"$p(-\psi)$")
	ax.vlines([-2*np.pi,-np.pi,0,+np.pi,+2*np.pi], *ax.get_ylim())
	ax.set_xlabel(r"$\psi$", fontsize=fsa)
	ax.set_ylabel(r"$p(\psi)$", fontsize=fsa)
	ax.grid(); ax.legend(loc="upper left", fontsize=fsl)
	## --------------------------------------------------------------------
	## Angles	
	ax = axs[1]
	Ecp = np.trapz(np.trapz(rho * np.cos(pp), etap, axis=2)*etar, etar, axis=1)*2*np.pi / (Q+(Q==0))
	Esp = np.trapz(np.trapz(rho * np.sin(pp), etap, axis=2)*etar, etar, axis=1)*2*np.pi / (Q+(Q==0))
	lEcp = ax.plot(r, Ecp, label=r"$\langle\cos\psi\rangle(r)$")
	lEsp = ax.plot(r, Esp, label=r"$\langle\sin\psi\rangle(r)$")
	ax.set_ylim(top=np.ceil(Ecp.max()))
	ax.set_xlabel(r"$r$", fontsize=fsa)
	ax.set_ylabel(r"Pointing", fontsize=fsa)
	ax.grid(); ax.legend(loc="upper left", fontsize=fsl)
	## --------------------------------------------------------------------
	## Force (assume dlin)	
	ax = axs[2]
	Eecp = np.trapz(np.trapz(rho * ee*np.cos(pp), etap, axis=2)*etar, etar, axis=1)*2*np.pi / (Q+(Q==0))
	Eesp = np.trapz(np.trapz(rho * ee*np.sin(pp), etap, axis=2)*etar, etar, axis=1)*2*np.pi / (Q+(Q==0))
	lEecp = ax.plot(r, Eecp, label=r"$\langle\eta\,\cos\psi\rangle(r)$")
	lEesp = ax.plot(r, Eesp, label=r"$\langle\eta\,\sin\psi\rangle(r)$")
	f = -np.hstack([np.linspace(r[0]-S,0.0,Sind),np.zeros(Rind-Sind),np.linspace(0,r[-1]-R,r.size-Rind)])	## Assume dlin
	ax.plot(r, -f, "k--", label=r"$-f(r)$")
	ax.set_ylim(top=np.ceil(Eecp.max()))
	ax.set_xlabel(r"$r$", fontsize=fsa)
	ax.set_ylabel(r"Force", fontsize=fsa)
	ax.grid(); ax.legend(loc="upper left", fontsize=fsl)
	## --------------------------------------------------------------------
	fig.suptitle(r"$\psi$ diagnostics. $\alpha=%.1f$, $R=%.1f$, $S=%.1f$"%(a,R,S), fontsize=fst)
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
