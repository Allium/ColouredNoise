me0 = "LE_PDF3D"

import numpy as np
import scipy.interpolate, scipy.ndimage, scipy.optimize
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
		plot_pdf3D(args[0], nosave, verbose)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		if verbose: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pdf3D(histfile, nosave, verbose)
			plt.close()
	else: raise IOError, me+"Check input."
	
	if verbose: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return

##=============================================================================
def plot_pdf3D(histfile, nosave, vb):
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
	r = 0.5*(rbins[1:]+rbins[:-1])
	etar = 0.5*(erbins[1:]+erbins[:-1])

	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	try:				H = H.sum(axis=2)	## If old file
	except ValueError: 	pass
	H /= np.trapz(np.trapz(H,etar,axis=1),r,axis=0)
	## To get probability density rather than probability
	rho = H / ( (2*np.pi)**2.0 * reduce(np.multiply, np.ix_(r,etar)) )
	
	## Resample
	npoints = 100
	x, y = np.linspace(r[0],r[-1],npoints), np.linspace(etar[0],etar[-1],npoints)
	X, Y = np.meshgrid(x, y)
#	Z = scipy.interpolate.RectBivariateSpline(r,etar,rho, s=0)(x,y,grid=True).T	## Slower
	Z = scipy.ndimage.interpolation.zoom(rho,[float(npoints)/r.size,float(npoints)/etar.size],order=1).T
	
	## Smooth
	Zsm = scipy.ndimage.gaussian_filter(Z, sigma=2.0, order=0, mode="nearest")

#	x, y = r, etar
#	X, Y = np.meshgrid(x, y)
#	Z = rho.T

	## Plotting
	fig = plt.figure()
	ax = fig.gca(projection="3d")

	## 3D contour plot
	ax.plot_surface(X, Y, Zsm, rstride=5, cstride=5, alpha=0.2, antialiased=True)
	
	## 2D contours
	xoff, yoff, zoff = X.min()-0.1*X.max(), -0.1*Y.max(), -0.1*Z.max()
	ax.contourf(X, Y, Zsm, zdir='z', offset=zoff,	cmap=cm.coolwarm, antialiased=True)	## 2D projection
	ax.contourf(X, Y, Zsm, zdir='x', offset=xoff,	cmap=cm.coolwarm, antialiased=True)	## p(eta)
#	ax.contourf(X, Y, Z, zdir='y', offset=yoff,	cmap=cm.coolwarm, antialiased=True)	## p(r) choppy
	ax.contourf(X, Y, Zsm, zdir='y', offset=yoff, cmap=cm.coolwarm, antialiased=True)	## p(r) smoothed

	## Plot smoothed p(r) envelope
	Q = np.trapz(Z.T*2*np.pi*y,y,axis=1)
	ax.plot(x, yoff*np.ones(y.shape),Q/Q.max()*Zsm.max(), "r--",lw=3)	## p(r) envelope
	E = np.trapz(Zsm*2*np.pi*y,y,axis=1)

	## Fit p(eta)
	fitfunc = lambda x, B, b: B*a*b/(2*np.pi)*np.exp(-0.5*a*b*y*y)
	fit = scipy.optimize.curve_fit(fitfunc, y, E, p0=[+1.0,+1.0])[0]
	ax.plot(xoff*np.ones(x.size), y, fitfunc(y,*fit)*Zsm.max()/E.max(), "g--", lw=3, zorder=2)
	
	## Wall
	if S>0:	ax.plot(S*np.ones(x.size), y, 0.8*zoff*np.ones(x.size), "g--", lw=3, zorder=2)
	if S>0:	ax.plot(R*np.ones(x.size), y, 0.8*zoff*np.ones(x.size), "g--", lw=3, zorder=2)

	## Accoutrements
	ax.set_zlim(zoff,ax.get_zlim()[1])
	ax.set_ylim(yoff,ax.get_ylim()[1])
	ax.elev = 30
	ax.azim = 45
	
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
