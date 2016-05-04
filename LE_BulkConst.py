import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os, optparse, glob
from LE_Utils import filename_pars
from LE_Utils import force_1D_const, force_1D_lin
from LE_Pressure import pressure_x
from LE_SPressure import Hr_norm, pdf_WN, plot_wall
from LE_SBS import force_const, force_lin, force_lico, force_dcon, force_dlin


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
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	vb = opt.verbose
	plotall = opt.plotall
		
	if os.path.isfile(args[0]):
		plotfile = plot_file(args[0])
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		plotfile = []
		for histfile in glob.glob(args[0]+"/BHIS*.npy"):
			plotfile += [plot_file(histfile)]
			plt.close()
	if os.path.isdir(args[0]):
		plotfile = plot_dir(args[0])
	
	if vb:	print me+"Figure saved to",plotfile
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile):
	## CALCULATIONS
	x, Q, e2E, c1, p, pars = bulk_const(histfile)
	ftype = pars["ftype"]
	ord = "r" if pars["geo"] == "CIR" else "x"
	## PLOTTING
	fig = plt.figure()
	fpars = [pars["X"]] if pars["geo"] is "1D" else [pars["R"],pars["S"]]
	plot_wall(plt.gca(), ftype, fpars, x)
	refpoint = Q.shape[0]/2 if ftype is "dcon" or ftype is "dlin" else 0
	plt.plot(x,Q/Q[refpoint],label="$\\rho("+ord+")$")
	plt.plot(x,e2E/e2E[refpoint],label="$\\langle\\eta^2\\rangle("+ord+")$")
	plt.plot(x,c1/c1[refpoint],label="$\\rho\\cdot\\langle\\eta^2\\rangle$")
	plt.xlim(left=x[0],right=10.0)
	plt.ylim(bottom=0.0,top=4.0)
	#plt.suptitle("Bulk Constant. $\\alpha = "+str(pars["a"])+"$.",fontsize=fst)
	plt.xlabel("$"+ord+"$",fontsize=fsa)
	plt.ylabel("Variable divided by first value",fontsize=fsa)
	plt.grid()
	plt.legend(loc="upper left",fontsize=fsl+2)
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+".png"
	plt.savefig(plotfile)
	return plotfile
	
##=============================================================================
def plot_dir(histdir):

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
		H = H.T
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		## r="x" for convenience
		x = 0.5*(rbins[1:]+rbins[:-1])
		eta = 0.5*(erbins[1:]+erbins[:-1]) ## Radial
		## Normalise probability
		H /= (H.sum()*np.outer(np.diff(erbins),np.diff(rbins)))
		## Marginalise over eta turn into density
		Q = sp.integrate.simps(H, x=eta, axis=0) / x
		## To get probability density rather than probability
		P = H / np.outer(eta,x)[::-1]
		## Force
		if ftype == "const":	force = force_const(x,x,x*x,R,R*R)
		elif ftype == "lin":	force = force_lin(x,x,x*x,R,R*R)
		elif ftype == "lico":	force = force_lico(x,x,x*x,R,R*R,g)
		elif ftype == "dcon":	force = force_dcon(x,x,x*x,R,R*R,S,S*S)
		elif ftype == "dlin":	force = force_dlin(x,x,x*x,R,R*R,S,S*S)
		p = -sp.integrate.simps(force*Q, x=x)
		e2E = sp.integrate.simps(((P/Q).T*(eta*eta)).T, x=eta, axis=0)
		c1 = Q*e2E
		
	if 0:
		print "eta pdfs"
		for i in range(0,50,2):
			plt.plot(eta, H[:,i]/Q[i])
		plt.show();exit()
		
	
	return [x, Q, e2E, c1, p, pars]
	
##=============================================================================
if __name__ == "__main__":
	main()
