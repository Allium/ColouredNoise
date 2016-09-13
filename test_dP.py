import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
import os, optparse, glob, time
from LE_Utils import filename_pars
from LE_SPressure import force_dlin, pdf_WN, plot_wall


from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	Plot components of the bulk constant.
	Adapted from LE_BulkConst.py
	For dlin only so far.
	"""
	me = "test_dP: "
	t0 = time.time()
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('--nosave',
		dest="nosave", default=False, action="store_true")
	opt, args = parser.parse_args()
	showfig = opt.showfig
	nosave = opt.nosave
	
	if os.path.isfile(args[0]):
		plot_file(args[0], nosave)
	else: raise IOError, me+"Check input."
	
	print me+"Execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_file(histfile, nosave):
	"""
	"""
	me = "test_dP.plot_file: "
	
	## CALCULATIONS
	r, Q, e2E, pars = bulk_const(histfile)
	ftype = pars["ftype"]
	R,S = [pars["R"],pars["S"]]
	
	fr = force_dlin(r,r,R,S)
	
	## PLOT
	fig = plt.figure(); ax = fig.gca()
	ax.axvspan(S,R,color="yellow",alpha=0.2)
	y = Q
	ax.plot(r,y/np.abs(y).mean(),label="$Q$")
	y = y[::-1]
	ax.plot(r,y/np.abs(y).mean(),label="$Q$ flip")
	y = fr	
	# ax.plot(r,y/np.abs(y).mean(),label="$f_r$")
	y = fr * Q
	ax.plot(r,y/np.abs(y).mean(),label="$f_rQ$")
	y = -y[::-1]
	# ax.plot(r,y/np.abs(y).mean(),label="$f_rQ$ flip")
	y = e2E
	ax.plot(r,y/np.abs(y).mean(),label="$\\langle\\eta^2\\rangle$")
	y = -(fr*fr-e2E)
	ax.plot(r,y/np.abs(y).mean(),label="$\\langle\\eta^2\\rangle-f_r^2$")
	y = -(fr*fr-e2E)*Q
	ax.plot(r,y/np.abs(y).mean(),label="$(\\langle\\eta^2\\rangle-f_r^2)Q$")
	y = y[::-1]
	ax.plot(r,y/np.abs(y).mean(),label="$(\\langle\\eta^2\\rangle-f_r^2)Q$ flip")
	# # y = np.diff((fr*fr-e2E)*Q)/np.diff(r)
	# # y = sp.ndimage.filters.gaussian_filter(y,5,order=0)
	# y = sp.ndimage.gaussian_filter1d(-(fr*fr-e2E)*Q,10,order=1)
	# ax.plot(r,y/np.abs(y).mean(),label="$\\partial_r((\\langle\\eta^2\\rangle-f_r^2)Q)$")
	
	## ATTRIBUTES
	ax.set_xlim(left=r[0],right=r[-1])
	ax.set_xlabel("$r$",fontsize=fsa)
	ax.set_ylabel("Rescaled variable",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl+2)
	fig.suptitle("$\\alpha = "+str(pars["a"])+"$.",fontsize=fst)
	
	fig.savefig(histfile[:-4]+".jpg")
	
	return
	
##=============================================================================
##=============================================================================

def bulk_const(histfile):

	pars = filename_pars(histfile)
	[a,X,R,S,D,lam,nu,ftype,geo] = [pars[key] for key in ["a","X","R","S","D","lam","nu","ftype","geo"]]

	H = np.load(histfile)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
			
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
	## Marginalise over eta turn into density
	Q = np.trapz(np.trapz(H,etap,axis=2),etar,axis=1) / (2*np.pi*r)
	
	## 3D arrays of etar and etap
	ETAR = etar[np.newaxis,:,np.newaxis].repeat(H.shape[0],axis=0).repeat(H.shape[2],axis=2)
	ETAP = etap[np.newaxis,np.newaxis,:].repeat(H.shape[0],axis=0).repeat(H.shape[1],axis=1)
	## Calculate averages
	Qp = Q+(Q==0) ## Avoid /0 warning (numerator is 0 anyway)
	er2E = np.trapz(np.trapz(H*ETAR*ETAR, etap, axis=2), etar, axis=1) / (2*np.pi*r*Qp)
	ep2E = np.trapz(np.trapz(H*ETAP*ETAP, etap, axis=2), etar, axis=1) / (2*np.pi*r*Qp)
	e2E = er2E*er2E*(1.0+ep2E*ep2E)
	e2E[np.isnan(e2E)] = 0.0
		
	return [r, Q, e2E, pars]
	
##=============================================================================
if __name__ == "__main__":
	main()
