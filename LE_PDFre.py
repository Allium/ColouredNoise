me0 = "LE_PDFre"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import pyplot as plt

from LE_Utils import filename_par
from LE_Utils import fs
fsa,fsl,fst = fs
from LE_SPressure import plot_wall

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

## ============================================================================

def main():
	"""
	Plot the marginalised densities Q(r) and q(eta).
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
		plot_pdfs(args[0], nosave, verbose)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		assert len(filelist)>1, me+"Check directory."
		if verbose: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pdfs(histfile, nosave, verbose)
			plt.close()
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_pdfs(histfile, nosave, vb):
	"""
	Calculate Q(r) and q(eta) from file and plot.
	"""
	
	me = me0+".plot_pdfs: "
	
	## Get pars from filename
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	ebins = bins["erbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	eta = 0.5*(ebins[1:]+ebins[:-1])
	
	## Wall indices
	Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()
	
	## Histogram
	H = np.load(histfile)
	try: H = H.sum(axis=2)
	except ValueError: pass
	H /= np.trapz(np.trapz(H,eta,axis=1),r,axis=0)
	
	## Spatial density
	Q = H.sum(axis=1)*(eta[1]-eta[0]) / (2*np.pi*r)
	## Force density
	q = H.sum(axis=0)*(r[1]-r[0]) / (2*np.pi*eta)
		
	##-------------------------------------------------------------------------
	
	fig, axs = plt.subplots(2,1)
	
	## Spatial density
	ax = axs[0]
	
	## Data
	ax.plot(r, Q, label=r"Simulation")
	
	## Gaussian
	if R==S:
		## Can't be bothered with normalisation
		gr = np.exp(-0.5*(a+1)*(r-R)**2.0)
		gr /= np.trapz(2*np.pi*r * gr, r)
		ax.plot(r, gr, label=r"$G\left(R, \frac{1}{\alpha+1}\right)$")
	
	## Potential
	if "_DL_" in histfile:
		ax.plot(r, (r-R)**2 * Q.max()/((r-R)**2).max(), "k--", label=r"$U(r)$")
	
	ax.set_xlabel(r"$r$", fontsize=fsa)
	ax.set_ylabel(r"$Q(r)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right", fontsize=fsl).get_frame().set_alpha(0.5)
	
	##-------------------------------------------------------------------------
	
	## Force density
	ax = axs[1]
	
	## Data
	ax.plot(eta, q, label=r"Simulation")
	
	## Gaussian
	ax.plot(eta, a/(2*np.pi)*np.exp(-0.5*a*eta**2.0), label=r"$G\left(0, \frac{1}{\alpha}\right)$")
	
	ax.set_xlabel(r"$\eta$", fontsize=fsa)
	ax.set_ylabel(r"$q(\eta)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right", fontsize=fsl).get_frame().set_alpha(0.5)
	
	##-------------------------------------------------------------------------
	
	fig.tight_layout()
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFre"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
	
	return
	
	
##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
