me0 = "test_RealPressure"

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

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## ============================================================================

def main():
	"""
	Adapted from LE_BulkConstnat. In notes 30/10/2016 derived an expression for
	pressure that depends on non-bulk integral. Test this: make a plot of P(a).
	Working with _psi files fo 28-30/10/2016.
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
	verbose = opt.verbose
		
	if os.path.isdir(args[0]):
		plot_dir(args[0], nosave, searchstr, verbose)
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

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
		[x, BC, p, pars] = bulk_const(histfile)
		if vb: print me+"File %i of %i: BC calculation %.2g seconds"%(i+1,numfiles,time.time()-t0)
		A[i] = pars["a"]
		P[i] = p
			
		fpars = [pars["R"],pars["S"],pars["lam"],pars["nu"]]
		Sind, Rind = np.abs(x-pars["S"]).argmin(), np.abs(x-pars["R"]).argmin()
		C[i] = BC[:Rind].mean() if (Sind==0 and Rind>0) else BC[Rind]	## Not fully functional
			
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
	plotfile = histdir+"/RPpol_Pa_R"+str(fpars[0])+"_S"+str(fpars[1])+".jpg"
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
	
	assert "_psi" in histfile, me+"Must use _psi file."
	
	H = np.load(histfile)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	
	## Circular sim
	if "CIR" in geo:
		
		## Space and load histogram
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		etar = 0.5*(erbins[1:]+erbins[:-1])
		epbins = bins["epbins"]
		etap = 0.5*(epbins[1:]+epbins[:-1])
		
		## Spatial arrays with dimensions commensurate to rho
		rr = r[:,np.newaxis,np.newaxis]
		ee = etar[np.newaxis,:,np.newaxis]
		pp = etap[np.newaxis,np.newaxis,:]
		dV = (r[1]-r[0])*(etar[1]-etar[0])*(etap[1]-etap[0])	## Assumes regular grid

		## Wall indices
		Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()
	
		## --------------------------------------------------------------------
		
		## Normalise histogram and convert to density
		H /= H.sum()*dV
		rho = H / ( (2*np.pi)**2.0 * rr*ee )
		
		## Marginalise over eta and calculate BC
		## Radial density
		Q = np.trapz(np.trapz(rho, etap, axis=2)*etar, etar, axis=1) * 2*np.pi
		
		## Bulk constant
		f = -np.hstack([np.linspace(S-r[0],0.0,Sind),np.zeros(Rind-Sind),np.linspace(0.0,r[-1]-R,r.size-Rind)])
		## <\eta^2\cos^2\psi>Q
		e2c2Q = np.trapz(np.trapz(rho * np.cos(pp)*np.cos(pp), etap, axis=2)*etar*etar * 2*np.pi*etar, etar, axis=1)
		e2s2Q = np.trapz(np.trapz(rho * np.sin(pp)*np.sin(pp), etap, axis=2)*etar*etar * 2*np.pi*etar, etar, axis=1)
		## <\eta^2>Q
#		e2Q = np.trapz(np.trapz(rho, etap, axis=2)*etar*etar * 2*np.pi*etar, etar, axis=1)
						
		## -\int_{bulk}^{\infty} (2<\eta^2\cos^2\psi>-<\eta^2>-f^2)*Q/r' dr'
		intgl = -sp.integrate.cumtrapz(((e2c2Q-e2s2Q-f*f*Q)/r)[::-1], r, axis=0, initial=0.0)[::-1]
		if S!=0.0:	intgl -= intgl[(Rind+Sind)/2]
		BC = e2c2Q + intgl
		
		## --------------------------------------------------------------------
		## Plot "bulk constant" components
		if 1:
			zoom = False
			t0 = time.time()
			fig = plt.figure(); ax = fig.gca()
			ax.plot(r,-f/np.abs(f).max(),"k--",label=r"$-f(r)$")
			ax.plot(r,Q/np.abs(Q).max(),":",label=r"$Q(r)$")
			ax.plot(r,e2Q/np.abs(e2Q).max(), label=r"$\langle\eta^2\rangle (r)Q(r)$")
			ax.plot(r,e2c2Q/np.abs(e2c2Q).max(), label=r"$\langle\eta^2 \cos^2\psi\rangle (r)Q(r)$")
			ax.plot(r,intgl/np.abs(intgl).max(),
				label=r"$-\int_r^\infty\frac{1}{r^\prime}\left(\langle\eta^2\cos^2\psi\rangle-\langle\eta^2\rangle-f^2\right)Q\,dr^\prime$")
			ax.plot(r,BC/np.abs(BC).max(), label=r"$P(r_{\rm bulk})/\alpha$")
			ax.axvline(S, color="k", linewidth=1); ax.axvline(R, color="k", linewidth=1)
			if S!=R:
				ax.axvspan(S,R, color="y", alpha=0.1)
				if zoom:	ax.set_ylim(np.around((np.array([-0.1,+0.1])+BC[Sind:Rind].mean()/np.abs(BC).max()),1))
			ax.set_xlabel(r"$r$", fontsize=fsa)
			ax.set_ylabel(r"Rescaled variable", fontsize=fsa)
			ax.legend(loc="best", fontsize=12).get_frame().set_alpha(0.5)
			ax.grid()
			ax.set_title(r"$a=%.1f, R=%.1f, S=%.1f$"%(a,R,S), fontsize=fst)
			plotfile = os.path.dirname(histfile)+("/RPpol_a%.1f_R%.1f_S%.1f"+"_zoom"*zoom+".jpg")%(a,R,S)
			fig.savefig(plotfile)
			print me+"Figure saved",plotfile
			print me+"BC components plot:",round(time.time()-t0,1),"seconds."
			plt.show()
			plt.close()
			exit()
		## --------------------------------------------------------------------
		
		## Integral pressure calculation
		## Integrate from bulk to infinity (outer wall)
		p = +calc_pressure(r[Rind:],Q[Rind:],ftype,[R,S,lam,nu])	## For outer wall
		# p = -calc_pressure(r[:Sind],Q[:Sind],ftype,[R,S,lam,nu])	## For inner wall
				
	return [r, BC, p, pars]

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
