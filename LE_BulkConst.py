import numpy as np
from matplotlib import pyplot as plt
import os, optparse, glob
from LE_Utils import filename_pars
from LE_Utils import force_1D_const as force_x
from LE_Pressure import pressure_x
from LE_SPressure import Hr_norm, plot_wall


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
	# if os.path.isdir(args[0]):
		# plotfile = plot_dir(args[0])
	
	if vb:	print me+"Figure saved to",plotfile
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile):
	## CALCULATIONS
	x, Hx, e2E, c1, p, pars = bulk_const(histfile)
	ord = "r" if pars["geo"] == "CIR" else "r"
	## PLOTTING
	fig = plt.figure()
	plot_wall(plt.gca(),pars["ftype"],x, (pars["X"] if pars["geo"] is "1D" else pars["R"]) )
	plt.plot(x,Hx/Hx[0],label="$Q("+ord+")$")
	plt.plot(x,e2E/e2E[0],label="$\\langle\\eta^2\\rangle("+ord+")$")
	plt.plot(x,c1/c1[0],label="$Q\\cdot\\langle\\eta^2\\rangle$")
	plt.xlim(left=x[0])
	plt.ylim(bottom=0.0,top=5.0)
	plt.suptitle("Bulk Constant. $\\alpha = "+str(pars["a"])+"$.")
	plt.xlabel("$"+ord+"$")
	plt.ylabel("Quantity divided by first value")
	plt.grid()
	plt.legend(loc="best")
	plotfile = os.path.dirname(histfile)+"/QEe2"+os.path.basename(histfile)[4:-4]+".png"
	plt.savefig(plotfile)
	return plotfile
	
##=============================================================================
def plot_dir(histdir):
	## CONSTRUCT CALCULATION ARRAYS
	A = []; X = []; C = []; P = []
	for histfile in np.sort(glob.glob(histdir+"*.npy")):
		pars = bulk_const(histfile)
		A += [pars[0]]
		X += [pars[1]]
		c = pars[4]
		C += [c[:c.shape[0]/2].mean()]
		P += [pars[5][-1]]
	A = np.array(A); X = np.array(X); C = np.array(C); P = np.array(P)
	## Theoretical calculation of white noise pressure
	P_WN = 1/(1.0-np.exp(X[:,X.shape[1]/2]-X[:,-1])+X[:,X.shape[1]/2]-X[:,0])
	## PLOTTING
	plt.plot(A,P, "o-", label="$-\\int\\rho(x)\\phi(x)\\,{\\rm d}x$")
	plt.plot(A,C*A, "o-", label="$\\alpha Q\\langle\\eta^2\\rangle$")
	plt.plot(A,P_WN,"--",label="WN theory")
	plt.suptitle("Pressure as Function of $\\alpha$")
	plt.xlabel("$\\alpha$")
	plt.grid()
	plt.legend()
	plotfile = histdir+"/QEe2_P_.png"
	plt.savefig(plotfile)
	return plotfile
	
##=============================================================================
##=============================================================================

def bulk_const(histfile):

	pars = filename_pars(histfile)
	[a,X,R,D,ftype,geo] = [pars[key] for key in ["a","X","R","D","ftype","geo"]]

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
		Hx = np.trapz(H,x=eta,axis=0)
		force = force_x(x,X,D)
		p = pressure_x(force,Hx,x)
		
	## Circular sim
	elif geo == "CIR":
		H = H.T
		## r->x for convenience
		rbins = bins["rbins"]
		erbins = bins["erbins"]
		x = 0.5*(rbins[1:]+rbins[:-1])
		eta = 0.5*(erbins[1:]+erbins[:-1]) ## Radial
		## To get probability density rather than probebility
		H /= np.meshgrid(x,eta)[0]#np.multiply(*np.meshgrid(x,eta))
		## Marginalise over eta
		Hx = np.trapz(H,x=eta,axis=0)
		Hx = Hr_norm(Hx,x,R)
		## Force
		force = 0.5*(np.sign(R-x)-1) * ((x-R) if ftype is "linear" else 1)
		p = -(force*Hx).cumsum() * (x[1]-x[0])
		
	if 0:
		print "eta pdfs"
		for i in range(0,50,2):
			plt.plot(eta, H[:,i]/Hx[i])
		plt.show();exit()
		
	e2E = np.trapz(((H/Hx).T*(eta*eta)).T,x=eta,axis=0)
	c1 = Hx*e2E
	
	return [x, Hx, e2E, c1, p, pars]
	
##=============================================================================
if __name__ == "__main__":
	main()