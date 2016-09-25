import numpy as np
import scipy as sp
from scipy.integrate import simps
from matplotlib import pyplot as plt
import os, optparse, glob, time
from LE_Utils import filename_pars, filename_par
from LE_Utils import force_1D_const, force_1D_lin
from LE_Pressure import pressure_x
innerwall = False
if innerwall:
	from LE_inSPressure import calc_pressure, pdf_WN, plot_wall
else:
	from LE_SPressure import calc_pressure, pdf_WN, plot_wall


from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	Plot the bulk constant <eta^2>Q as a function of r for a single file, or...
	"""
	me = "LE_BulkConst: "
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
	me = "LE_BulkConst.plot_file: "

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
	# fig.suptitle("Bulk Constant. $\\alpha = "+str(pars["a"])+"$.",fontsize=fst)
	
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
	me = "LE_BulkConst.plot_dir: "
	
	dirpars = filename_pars(histdir)
	geo = dirpars["geo"]
	ftype = dirpars["ftype"]
	assert ftype[0]!="d", me+"Functionality not available."
	
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
		
		if geo == "1D":
			X[i] = pars["X"]
			Xidx = np.argmin(np.abs(x-X[i]))
			force = 0.5*(np.sign(X[i]-x)-1)* ((x-X[i]) if ftype is "lin" else 1)
			P_WN[i] = -(force*pdf_WN(x,R[i],ftype)).sum()*(x[1]-x[0])
			C[i] = c1[:widx].mean()
			
		elif geo == "CIR":
			fpars = [pars["R"],pars["S"],pars["lam"],pars["nu"]]
			Ridx, Sidx = np.abs(x-fpars[0]).argmin(), np.abs(x-fpars[1]).argmin()
			r_WN = np.linspace(x[0],x[-1],2*x.size+1)
			P_WN[i] = calc_pressure(r_WN,pdf_WN(r_WN,fpars,ftype),ftype,fpars)
			C[i] = c1[Sidx+10:Ridx-10].mean()
	
	## NORMALISE
	P /= P_WN
	C /= P_WN
	
	## PLOTTING
	fig = plt.figure(); ax = fig.gca()
	ax.plot(A,P, "o-", label="$-\\int\\rho(x)\\phi(x)\\,{\\rm d}x$")
	ax.plot(A,C*A, "o-", label="$\\alpha Q\\langle\\eta^2\\rangle$")
	ax.set_xlabel("$\\alpha$")
	ax.set_ylabel("$P$")
	ax.grid()
	ax.legend()
	fig.suptitle("Pressure normalised by WN result")
	
	plotfile = histdir+"/QEe2_P_.png"
	if not nosave:
		fig.savefig(plotfile)
		if vb: print me+"Figure saved to",plotfile
		
	return plotfile
	
##=============================================================================
##=============================================================================

def bulk_const(histfile):

	try:
		pars = filename_pars(histfile)
		[a,X,R,S,D,lam,nu,ftype,geo] = [pars[key] for key in ["a","X","R","S","D","lam","nu","ftype","geo"]]
	except:
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
		
		## Space
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		epbins = bins["epbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		etar = 0.5*(erbins[1:]+erbins[:-1])
		etap = 0.5*(epbins[1:]+epbins[:-1])
		
		## Probability
		## Normalise
		H /= np.trapz(np.trapz(np.trapz(H,etap,axis=2),etar,axis=1),r,axis=0)
		## Marginalise over eta turn into radial density
		Q = np.trapz(np.trapz(H,etap,axis=2),etar,axis=1) / (2*np.pi*r)
		## To get probability density rather than probability
		rho = H / reduce(np.multiply, np.ix_(r,etar,etap))
		## Normalise so Q=1 in the bulk
		# if innerwall:	fac = Q[-r.size/6:].mean()
		# else:			fac = Q[:r.size/6].mean()
		# rho/=fac; H/=fac; Q/=fac
				
		## Calculations
		p = calc_pressure(r,Q,ftype,[R,S,lam,nu])
		
		## 3D arrays of etar and etap
		ETAR = etar[np.newaxis,:,np.newaxis].repeat(H.shape[0],axis=0).repeat(H.shape[2],axis=2)
		ETAP = etap[np.newaxis,np.newaxis,:].repeat(H.shape[0],axis=0).repeat(H.shape[1],axis=1)
		## Calculate averages
		Qp = Q+(Q==0) ## Avoid /0 warning (numerator is 0 anyway)
		er2E = np.trapz(np.trapz(H*ETAR*ETAR, etap, axis=2), etar, axis=1) / (2*np.pi*r*Qp)
		# er2E = np.trapz(np.trapz(rho*ETAR*ETAR, etap, axis=2), etar, axis=1) / (Qp)
		ep2E = np.trapz(np.trapz(H*ETAP*ETAP, etap, axis=2), etar, axis=1) / (2*np.pi*r*Qp)	### ???
		e2E = er2E*er2E#*(1.0+ep2E*ep2E)
		e2E[np.isnan(e2E)] = 0.0
		## Bulk constant
		c1 = Q*e2E
		c1 = sp.ndimage.filters.gaussian_filter(c1,1,order=0)
	
	Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()
	# print [a,S,R],"\t",round(c1[Sind:Rind+1].mean(),5)
	
	try: x = r
	except UnboundLocalError: pass
	
	return [r, Q, e2E, c1, p, pars]
	
##=============================================================================
if __name__ == "__main__":
	main()
