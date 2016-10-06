me0 = "LE_PDF3D"

import numpy as np
import scipy.interpolate
from sys import argv
import os, optparse, glob, time

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt

from LE_Utils import filename_par

import warnings
warnings.filterwarnings("ignore",category=FutureWarning)

"""
TO DO
Gaussian fits
"""

def input():
	"""
	Read command-line arguments and decide which plot to make.
	"""

	me = me0+".input: "
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
		plot_pdf3D_file(args[0], nosave, verbose)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		if verbose: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pdf3D_file(histfile, nosave, verbose)
			plt.close()
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_pdf3D_file(histfile, nosave, vb):
	"""
	Read in data for a single file and plot 3D PDF.
	"""
	me = me0+"plot_pdf3D_file: "

	## Get pars from filename
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
			
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins  = bins["rbins"]
	erbins = bins["erbins"]
	epbins = bins["epbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	etar = 0.5*(erbins[1:]+erbins[:-1])
	etap = 0.5*(epbins[1:]+epbins[:-1])

	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	H /= np.trapz(np.trapz(np.trapz(H,etap,axis=2),etar,axis=1),r,axis=0)
	try: H = H.sum(axis=2)
	except ValueError: pass
	## To get probability density rather than probability
	rho = H / ( (2*np.pi)**2.0 * reduce(np.multiply, np.ix_(r,etar)) )

	## Create grids
	NGridPoints = 100
	x, y = np.linspace(r[0],r[-1],NGridPoints), np.linspace(etar[0],etar[-1],NGridPoints)
	X, Y = np.meshgrid(x, y)
	Z = scipy.interpolate.RectBivariateSpline(r,etar,rho)(x,y,grid=True)

	## Plotting
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	ax.plot_surface(X, Y, Z, rstride=5, cstride=5, alpha=0.2)
	ax.contourf(X, Y, Z, zdir='z', offset=-0.2*Z.max(), cmap=cm.coolwarm)
	ax.contourf(X, Y, Z, zdir='x', offset=0.0, 	cmap=cm.coolwarm)
	ax.contourf(X, Y, Z, zdir='y', offset=etar[-1], cmap=cm.coolwarm)

	## Fit / prediction
	# ax.plot(x, np.sqrt(a/(2*np.pi))*np.exp(-0.5*a*x*x), "g--", zdir="x")

	## Accoutrements
	ax.set_zlim(-0.2*Z.max(),ax.get_zlim()[1])
	ax.set_xlabel(r"$r$", 		fontsize=18)
	ax.set_ylabel(r"$\eta_r$", 	fontsize=18)
	ax.set_zlabel(r"$\rho$", 	fontsize=18)
	fig.suptitle(r"PDF in $r$-$\eta_r$ space. $\alpha="+str(a)+", R="+str(R)+", S="+str(S)+"$",
				fontsize=16)
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDF3D"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+": Figure saved to",plotfile

	return
	
##=============================================================================
if __name__ == "__main__":
	input()
