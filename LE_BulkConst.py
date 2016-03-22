import numpy as np
from matplotlib import pyplot as plt
from LE_Utils import filename_pars
from LE_Utils import FBW_soft as force_x
from LE_Pressure import pressure_x
import os, optparse, glob


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
		for histfile in glob.glob(args[0]+"*.npy"):
			plotfile += plot_file(histfile)
			plt.clf()
	if os.path.isdir(args[0]):
		plotfile = plot_dir(args[0])
	
	if vb:	print me+"Figure saved to",plotfile
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile):
	## CALCULATIONS
	a, x, Hx, e2E, c1, p = bulk_const(histfile)
	## PLOTTING
	plt.axvline(x[x.shape[0]/2-1:x.shape[0]/2+1].mean(),color="k")
	plt.plot(x,Hx/Hx[0],label="$Q(x)$")
	plt.plot(x,e2E/e2E[0],label="$\\langle\\eta^2\\rangle(x)$")
	plt.plot(x,c1/c1[0],label="$Q\\cdot\\langle\\eta^2\\rangle$")
	plt.xlim(left=x[0])
	plt.ylim(bottom=0.0,top=np.ceil((c1/c1[0]).max()))
	plt.suptitle("Bulk Constant. $\\alpha = "+str(a)+"$.")
	plt.xlabel("$x$")
	plt.ylabel("Quantity divided by first value")
	plt.grid()
	plt.legend()
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
	[a,X,D,dt] = [pars[key] for key in ["a","X","D","dt"]]

	H = np.load(histfile)
	# H[:,0] = H[:,1]

	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	ybins = bins["ybins"]

	x = 0.5*(xbins[1:]+xbins[:-1])
	eta = 0.5*(ybins[1:]+ybins[:-1])

	H /= np.trapz(np.trapz(H,x=x,axis=1),x=eta,axis=0)
	Hx = np.trapz(H,x=eta,axis=0)

	if 0:
		print "eta pdfs"
		for i in range(H.shape[1]/2):
			plt.plot(eta, H[:,i]/Hx[i])
		plt.show();exit()
	
	force = force_x(x,1.0,X,D)
	press = pressure_x(force,Hx,x)

	## Must normalise each eta pdf slice
	e2E = np.trapz(((H/Hx).T*(eta*eta)).T,x=eta,axis=0)
	c1 = Hx*e2E
	
	return a, x, Hx, e2E, c1, press
	
##=============================================================================
if __name__ == "__main__":
	main()