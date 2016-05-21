import numpy as np
import scipy as sp
from scipy.integrate import simps
from matplotlib import pyplot as plt
import os, optparse, glob
from LE_Utils import filename_pars
from LE_Utils import force_1D_const, force_1D_lin
from LE_SBS import force_const, force_lin, force_lico, force_dcon, force_dlin
from LE_Pressure import pressure_x
from LE_SPressure import pdf_WN, plot_wall


from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	Plot the constant.
	"""
	me = "LE_BulkConst: "
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	nosave = opt.nosave
	vb = opt.verbose
	plotall = opt.plotall
		
	if os.path.isfile(args[0]):
		plotfile = plot_file(args[0], nosave)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		plotfile = []
		for histfile in glob.glob(args[0]+"/BHIS*.npy"):
			plotfile += [plot_file(histfile, nosave)]
			plt.close()
	if os.path.isdir(args[0]):
		plotfile = plot_dir(args[0], nosave)
	
	if (vb and not nosave):	print me+"Figure saved to",plotfile
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile,nosave):
	## CALCULATIONS
	x, Q, e2E, c1, p, pars = bulk_const(histfile)
	ftype = pars["ftype"]
	fpars = [pars["X"]] if pars["geo"] is "1D" else [pars["R"],pars["S"]]
	ord = "r" if pars["geo"] == "CIR" else "x"
	## PLOT
	fig = plt.figure()
	plot_wall(plt.gca(), ftype, fpars, x)
	refpoint = Q.shape[0]/2 if (ftype is "dcon" or ftype is "dlin") else 0 ## fix
	plt.plot(x,Q/Q.mean(),label="$\\rho("+ord+")$")
	plt.plot(x,e2E/e2E.mean(),label="$\\langle\\eta^2\\rangle("+ord+")$")
	plt.plot(x,c1/c1.mean(),label="$\\rho\\cdot\\langle\\eta^2\\rangle$")
	plt.xlim(left=x[0],right=x[-1])
	ymax = 3.0*np.median((c1/c1.mean())[:np.abs(fpars[0]-x).argmin()])
	plt.ylim(bottom=0.0,top=ymax)
	plt.suptitle("Bulk Constant. $\\alpha = "+str(pars["a"])+"$.",fontsize=fst)
	plt.xlabel("$"+ord+"$",fontsize=fsa)
	plt.ylabel("Variable divided by first value",fontsize=fsa)
	plt.grid()
	plt.legend(loc="upper left",fontsize=fsl+2)
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+".png"
	if not nosave:	plt.savefig(plotfile)
	return plotfile
	
##=============================================================================
def plot_dir(histdir,nosave):

	dirpars = filename_pars(histdir)
	geo = dirpars["geo"]
	ftype = dirpars["ftype"]
	
	## CONSTRUCT CALCULATION ARRAYS
	A = []; X = []; R = []; C = []; P = []; P_WN = []
	for i,histfile in enumerate(np.sort(glob.glob(histdir+"*_R5.0_*.npy"))):
		[x, Hx, e2E, c1, p, pars] = bulk_const(histfile)
		A += [pars["a"]]
		if geo == "1D":
			X += [pars["X"]]
			widx = np.argmin(np.abs(x-X[i]))
			force = 0.5*(np.sign(X[i]-x)-1)* ((x-X[i]) if ftype is "linear" else 1)
			P_WN += [-(force*pdf_WN(x,R[i],ftype)).sum()*(x[1]-x[0])]
		elif geo == "CIR":
			R += [pars["R"]]
			widx = np.argmin(np.abs(x-R[i]))
			force = 0.5*(np.sign(R[i]-x)-1) * ((x-R[i]) if ftype is "linear" else 1)
			P_WN += [-(force*pdf_WN(x,[R[i]],ftype)).sum()*(x[1]-x[0])]
		C += [c1[:widx].mean()]
		P += [p]
	A = np.array(A); X = np.array(X); R = np.array(R); C = np.array(C)
	P = np.array(P); P_WN = np.array(P_WN)
	
	## NORMALISE
	P /= P_WN
	C /= P_WN
	
	## PLOTTING
	plt.plot(A,P, "o-", label="$-\\int\\rho(x)\\phi(x)\\,{\\rm d}x$")
	plt.plot(A,C*A, "o-", label="$\\alpha Q\\langle\\eta^2\\rangle$")
	plt.suptitle("Pressure normalised by WN result")
	plt.xlabel("$\\alpha$")
	plt.ylabel("$P$")
	plt.grid()
	plt.legend()
	plotfile = histdir+"/QEe2_P_.png"
	# plt.savefig(plotfile)
	return plotfile
	
##=============================================================================
##=============================================================================

def bulk_const(histfile):

	pars = filename_pars(histfile)
	[a,X,R,S,D,ftype,geo] = [pars[key] for key in ["a","X","R","S","D","ftype","geo"]]

	H = np.load(histfile)
	if len(H.shape)==4:
		H = H.sum(axis=1)
		import LE_Utils
		LE_Utils.save_data(histfile, H)
		print "summed and resaved\n"
		exit()
		
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	
	## 1D sim
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
		c1 = Q*e2E
		
	## Circular sim
	elif geo == "CIR":
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		epbins = bins["epbins"]
		## r="x" for convenience
		x = 0.5*(rbins[1:]+rbins[:-1])
		etar = 0.5*(erbins[1:]+erbins[:-1])
		etap = 0.5*(epbins[1:]+epbins[:-1])
		## Normalise probability
		H /= simps(simps(simps(H,etap,axis=2),etar,axis=1),x,axis=0)
		## Marginalise over eta turn into density
		Q = simps(simps(H,etap,axis=2),etar,axis=1) / x
		## To get probability density rather than probability
		# rho = (H.T / eta).T / x
		rho = H / reduce(np.multiply, np.ix_(x,etar,etap))
		## Force
		if ftype == "const":	force = force_const(x,x,x*x,R,R*R)
		elif ftype == "lin":	force = force_lin(x,x,x*x,R,R*R)
		elif ftype == "lico":	force = force_lico(x,x,x*x,R,R*R,g)
		elif ftype == "dcon":	force = force_dcon(x,x,x*x,R,R*R,S,S*S)
		elif ftype == "dlin":	force = force_dlin(x,x,x*x,R,R*R,S,S*S)
		
		## Calculations
		p = -simps(force*Q, x=x)
		
		ETAR = etar[np.newaxis,:,np.newaxis].repeat(H.shape[0],axis=0).repeat(H.shape[2],axis=2)
		ETAP = etap[np.newaxis,np.newaxis,:].repeat(H.shape[0],axis=0).repeat(H.shape[1],axis=1)
		er2E = simps(simps((H.T/Q).T*ETAR*ETAR, etap,axis=2), etar,axis=1)/x
		ep2E = simps(simps((H.T/Q).T*ETAP*ETAP, etap,axis=2), etar,axis=1)/x
		e2E = er2E*er2E+ep2E*ep2E
		e2E[np.isnan(e2E)] = 0.0
		c1 = Q*e2E
		c1 = sp.ndimage.filters.gaussian_filter(c1,1,order=0)
	
	return [x, Q, e2E, c1, p, pars]
	
##=============================================================================
if __name__ == "__main__":
	main()
