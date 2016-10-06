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
from LE_Utils import force_1D_const, force_1D_lin
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()
from LE_Pressure import pressure_x
from LE_SPressure import calc_pressure, pdf_WN, plot_wall



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
	x, Q, e2E, c1, p, pars = bulk_const(histfile)
	ftype = pars["ftype"]
	fpars = [pars["X"]] if pars["geo"] is "1D" else [pars["R"],pars["S"],pars["lam"],pars["nu"]]
	ord = "r" if pars["geo"] == "CIR" else "x"
	
	## PLOT
	fig = plt.figure(); ax = fig.gca()
	plot_wall(ax, ftype, fpars, x)
	ax.plot(x,Q/Q.mean(),label="$\\rho("+ord+")$")
	ax.plot(x,e2E/e2E.mean(),label="$\\langle\\eta^2\\rangle("+ord+")$")
	ax.plot(x,c1/c1.mean(),label="$\\rho\\cdot\\langle\\eta^2\\rangle$")
	
	ax.axvspan(pars["S"],pars["R"],color="yellow",alpha=0.2)
	
	## ATTRIBUTES
	ax.set_xlim(left=x[0],right=x[-1])
	if ftype[0]!="d":
		if innerwall:	ymax = 3.0*np.median((c1/c1.mean())[np.abs(fpars[0]-x).argmin():])
		else:			ymax = 3.0*np.median((c1/c1.mean())[:np.abs(fpars[0]-x).argmin()+1])
	else:	ymax = 3.0*np.median((c1/c1.mean())[np.abs(fpars[1]-x).argmin():np.abs(fpars[0]-x).argmin()+1])
	ax.set_ylim(bottom=0.0,top=ymax)
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
	
		[x, Hx, e2E, c1, p, pars] = bulk_const(histfile)
		A[i] = pars["a"]
		P[i] = p
		
		"""if geo == "1D":
			X[i] = pars["X"]
			Xidx = np.argmin(np.abs(x-X[i]))
			force = 0.5*(np.sign(X[i]-x)-1)* ((x-X[i]) if ftype is "lin" else 1)
			P_WN[i] = -(force*pdf_WN(x,R[i],ftype)).sum()*(x[1]-x[0])
			C[i] = c1[:widx].mean()"""
			
		if geo == "CIR":
			fpars = [pars["R"],pars["S"],pars["lam"],pars["nu"]]
			Ridx, Sidx = np.abs(x-fpars[0]).argmin(), np.abs(x-fpars[1]).argmin()
			C[i] = c1[Sidx+10:Ridx-10].mean() if Ridx!=Sidx else c1[max(0,Sidx-10):Ridx+10].mean()  ##MESS
			
	## SORT BY ALPHA
	srtidx = A.argsort()
	A = A[srtidx]; P = P[srtidx]; P_WN = P_WN[srtidx]; C = C[srtidx]
	
	## Calculate white noise pressure and pdf
	## Assume all files have same R & S. P_WN independent of alpha.
	r_WN = np.linspace(x[0],x[-1],2*x.size+1)
	Ridx_WN, Sidx_WN = np.abs(r_WN-fpars[0]).argmin(), np.abs(r_WN-fpars[1]).argmin()			
	p_WN = calc_pressure(r_WN,pdf_WN(r_WN,fpars,ftype),ftype,fpars,True)
	P_WN = p_WN[-1] - p_WN.min()	## For outer wall
#	P_WN = p_WN[0]  - p_WN.min()	## For inner wall
	
	## NORMALISE
	P /= P_WN
	C /= P_WN
	
	## FIT -- A*C -- fit in log coordinates
	fitfunc = lambda x, B, nu: B + nu*x
	fitBC = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(A*C), p0=[+1.0,-1.0])[0]
	if vb:	print me+": BC:  [B, nu] = ",fitBC.round(3)
	## FIT -- P_int
	fitP = sp.optimize.curve_fit(fitfunc, np.log(1+A), np.log(P), p0=[+1.0,-1.0])[0]
	if vb:	print me+": Int: [B, nu] = ",fitP.round(3)
	
	## PLOT DATA AND FIT
	fig = plt.figure(); ax = fig.gca()
	
	ax.plot(1+A, P, "o-", label=r"$-\int_{\rm bulk}^{\infty} fQ\,{\rm d}r$")
	ax.plot(1+A, A*C, "o-", label=r"$\alpha Q\langle\eta^2\rangle|_{\rm bulk}$")

	ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitP)), "b--", lw=1,
			label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitP[0]),fitP[1]))
	ax.plot(1+A, np.exp(fitfunc(np.log(1+A), *fitBC)), "g--", lw=1,
			label=r"$%.1g(1+\alpha)^{%.3g}$"%(np.exp(fitBC[0]),fitBC[1]))
	
	## ACCOUTREMENTS
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlim(right=1+A[-1])
	
	ax.set_xlabel(r"$\alpha+1$",fontsize=fsa)
	ax.set_ylabel(r"$P$",fontsize=fsa)
	ax.grid()
	ax.legend()
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
		c1 = Q*e2E"""
		
	## Circular sim
	if "CIR" in geo:
		
		## Space and load histogram
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		etar = 0.5*(erbins[1:]+erbins[:-1])
		## For old _phi files
		try:
			epbins = bins["epbins"]
			etap = 0.5*(epbins[1:]+epbins[:-1])	## Unneccessary
			H = H.sum(axis=2) * (etap[1]-etap[0])
		except KeyError:
			pass
		
		## Probability
		## Normalise
		H /= np.trapz(np.trapz(H,etar,axis=1),r,axis=0)
		## Marginalise over eta turn into radial density
		Q = np.trapz(H,etar,axis=1) / (2*np.pi*r)
		## To get probability density rather than probability
		rho = H / ( (2*np.pi)**2.0 * reduce(np.multiply, np.ix_(r,etar)) )
				
		## Conventional pressure calculation
		## For disc, integrate from zero; for annulus, integrate from bulk to infinity (outer wall)
		Ridx, Sidx = np.abs(r-R).argmin(), np.abs(r-S).argmin()
		p = +calc_pressure(r[Ridx:],Q[Ridx:],ftype,[R,S,lam,nu])	## For outer wall
#		p = -calc_pressure(r[:Sidx],Q[:Sidx],ftype,[R,S,lam,nu])	## For inner wall
		
		## 2D array of etar
		ETAR = etar[np.newaxis,:].repeat(H.shape[0],axis=0)
		## Calculate averages
		Qp = Q+(Q==0) ## Avoid /0 warning (numerator is 0 anyway)
		er2E = np.trapz(rho * etar * (etar*etar), etar, axis=1) / (Qp)	## 1 file
#		er2E = np.trapz(rho * (etar*etar), etar, axis=1) / (Qp)			## 2 P(a)
		e2E = er2E*er2E
		e2E[np.isnan(e2E)] = 0.0
		## Bulk constant
		c1 = Q*e2E * (2.0*np.pi)**2.0
#		c1 = sp.ndimage.filters.gaussian_filter(c1,1,order=0)
	
	try: x = r
	except UnboundLocalError: pass
	
	return [r, Q, e2E, c1, p, pars]
	
##=============================================================================
if __name__ == "__main__":
	main()
