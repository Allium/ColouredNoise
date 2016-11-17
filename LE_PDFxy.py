me0 = "LE_PDFxy"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import os, optparse, glob, time

if "SSH_TTY" in os.environ:
	print me0+": Using Agg backend."
	import matplotlib as mpl
	mpl.use("Agg")
from matplotlib import cm
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
	Plot the marginalised densities Q(x), qx(etax) and  qy(etay).
	Adapted from LE_PDFre.py.
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
	vb = opt.verbose
	
	## Plot file
	if os.path.isfile(args[0]):
		plot_pdf1d(args[0], nosave, vb)
		plot_pdf2d(args[0], nosave, vb)
	## Plot all files
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pdf1d(histfile, nosave, vb)
			plot_pdf2d(histfile, nosave, vb)
			plt.close()
	## Plot directory
	elif os.path.isdir(args[0]):
		plot_fitpars(args[0], searchstr, nosave, vb)
	else: raise IOError, me+"Check input."
	
	if vb: print me+"Total execution time",round(time.time()-t0,1),"seconds."
	if showfig:	plt.show()
	
	return


##=============================================================================
def plot_pdf1d(histfile, nosave, vb):
	"""
	Calculate Q(r) and q(eta) from file and plot.
	"""
	me = me0+".plot_pdf1d: "
	t0 = time.time()
	
	## Get pars from filename
	assert "_CAR_" in histfile, me+"Functional only for Cartesian geometry."
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	
	doQfit = (R==S and "_DL_" in histfile)
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	exbins = bins["exbins"]
	eybins = bins["eybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	etax = 0.5*(exbins[1:]+exbins[:-1])
	etay = 0.5*(eybins[1:]+eybins[:-1])
	
	## Wall indices
	Rind, Sind = np.abs(x-R).argmin(), np.abs(x-S).argmin()
	
	##-------------------------------------------------------------------------
	
	## Histogram
	H = np.load(histfile)
	rho = H / (H.sum() * (x[1]-x[0])*(etax[1]-etax[0])*(etay[1]-etay[0]))
	
	## Spatial density
	Qx = rho.sum(axis=2).sum(axis=1) * (etax[1]-etax[0])*(etay[1]-etay[0])
	## Force density
	qx = rho.sum(axis=2).sum(axis=0) * (x[1]-x[0])*(etay[1]-etay[0])
	qy = rho.sum(axis=1).sum(axis=0) * (x[1]-x[0])*(etax[1]-etax[0])
		
	##-------------------------------------------------------------------------
	## Fit
	gauss = lambda x, m, s2: 1/np.sqrt(2*np.pi*s2)*np.exp(-0.5*(x-m)**2/s2)
	
	if doQfit: fitQx = sp.optimize.curve_fit(gauss, x, Qx, p0=[R,1/np.sqrt(1+a)])[0]
	
	##-------------------------------------------------------------------------
	
	fig, axs = plt.subplots(2,1, figsize=(10,10))
	fig.canvas.set_window_title("1D PDFs")
	
	## Spatial density plot
	ax = axs[0]
	
	## Data
	ax.plot(x, Qx, label=r"Simulation")
	
	## Gaussian
	if doQfit:
		ax.plot(x, gauss(x,fitQx[0],1/(1+a)), "c-", label=r"$G\left(\mu, \frac{1}{\alpha+1}\right)$")
	
	## Potential and WN
	if "_DL_" in histfile:
		Ufn = lambda Dx: 0.5*Dx**2
		U = np.hstack([Ufn(S-x[:Sind]),np.zeros(Rind-Sind),Ufn(x[Rind:]-R)])
		ax.plot(x,np.exp(-U)/np.trapz(np.exp(-U),x),"r-", label="WN")
		ax.plot(x,U/U.max()*ax.get_ylim()[1],"k--",label="Potential")
	
	ax.set_xlim(left=x[0],right=x[-1])
	ax.set_xlabel(r"$x$", fontsize=fsa)
	ax.set_ylabel(r"$Q(x)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right", fontsize=fsl).get_frame().set_alpha(0.5)
		
	##-------------------------------------------------------------------------
	
	## Force density plot
	ax = axs[1]
	
	## Data
	ax.plot(etax, qx, label=r"Simulation $x$")
	ax.plot(etay, qy, label=r"Simulation $y$")
	
	## Gaussian
	ax.plot(etax, gauss(etax,0.0,1/a), "c-", label=r"$G\left(0, \frac{1}{\alpha}\right)$")
	
	ax.set_xlabel(r"$\eta$", fontsize=fsa)
	ax.set_ylabel(r"$q(\eta)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right", fontsize=fsl).get_frame().set_alpha(0.5)
	
	##-------------------------------------------------------------------------
	
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	fig.suptitle(r"PDFs in $r$ and $\eta$. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S), fontsize=fst)
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFxy1d"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)
	
	return
	
	
##=============================================================================
def plot_pdf2d(histfile, nosave, vb):
	"""
	Read in data for a single file and plot 3D PDF.
	"""
	me = me0+".plot_pdf2D: "
	t0 = time.time()

	## Get pars from filename
	assert "_CAR_" in histfile, me+"Functional only for Cartesian geometry."
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
		
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	exbins = bins["exbins"]
	eybins = bins["eybins"]
	x = 0.5*(xbins[1:]+xbins[:-1])
	etax = 0.5*(exbins[1:]+exbins[:-1])
	etay = 0.5*(eybins[1:]+eybins[:-1])
	
	X, EX, EY = np.meshgrid(x, etax, etay, indexing="ij")
	
	## Wall indices
	Rind, Sind = np.abs(x-R).argmin(), np.abs(x-S).argmin()
	
	##-------------------------------------------------------------------------
	
	## Histogram / density
	H = np.load(histfile)
	rho = H / (H.sum() * (x[1]-x[0])*(etax[1]-etax[0])*(etay[1]-etay[0]))
	
	## Prediction for when R=S and potential is quadratic
	pred = int(R==S and "_DL_" in histfile)
	if pred:
		Xc = 0.5*(R+S)	## To centre the prediction
		rhoP = a*(a+1)/(2*np.sqrt(2)*np.pi**1.5)*\
				np.exp(-0.5*(a+1)**2*(X-Xc)*(X-Xc)-0.5*a*(a+1)*EX*EX+a*(a+1)*(X-Xc)*EX-0.5*a*EY*EY)
		
	## ------------------------------------------------------------------------
	
	## Projections
	
	rhoxex = rho.sum(axis=2)
	rhoxey = rho.sum(axis=1)
	rhoexey = rho.sum(axis=0)
	
	if pred:
		rhoPxex = rhoP.sum(axis=2)
		rhoPxey = rhoP.sum(axis=1)
		rhoPexey = rhoP.sum(axis=0)
	
	## ------------------------------------------------------------------------
	
	## Plotting
	
	fig, axs = plt.subplots(3,1+pred, sharey=True, figsize=(10,10))
	fig.canvas.set_window_title("2D PDFs")
	
	plt.rcParams["image.cmap"] = "Greys"#"coolwarm"
	
	## ------------------------------------------------------------------------
	
	## x-etax
	
	ax = axs[0][0] if pred else axs[0]
	ax.contourf(x,etax,rhoxex.T)
	ax.axvline(R,c="k"); ax.axvline(S,c="k")
	
	ax.set_xlabel(r"$x$", fontsize=fsa)
	ax.set_ylabel(r"$\eta_x$", fontsize=fsa)
	ax.set_title(r"$\rho(x,\eta_x)$ data", fontsize=fsa)
	
	if pred:
		ax = axs[0][1]
		ax.contourf(x,etax,rhoPxex.T)
		ax.axvline(Xc,c="k")
	
		ax.set_xlabel(r"$x$", fontsize=fsa)
		ax.set_title(r"$\rho(x,\eta_x)$ prediction ($R=S$)", fontsize=fsa)
	
	## x-etay
	
	ax = axs[1][0] if pred else axs[1]
	ax.contourf(x,etay,rhoxey.T)
	ax.axvline(R,c="k"); ax.axvline(S,c="k")
	
	ax.set_xlabel(r"$x$", fontsize=fsa)
	ax.set_ylabel(r"$\eta_y$", fontsize=fsa)
	ax.set_title(r"$\rho(x,\eta_y)$ data", fontsize=fsa)
	
	if pred:
		ax = axs[1][1]
		ax.contourf(x,etay,rhoPxey.T)
		ax.axvline(Xc,c="k")
	
		ax.set_xlabel(r"$x$", fontsize=fsa)
		ax.set_title(r"$\rho(x,\eta_y)$ prediction ($R=S$)", fontsize=fsa)
	
	## etax-etay
	
	ax = axs[2][0] if pred else axs[2]
	ax.contourf(etax,etay,rhoexey.T)
	
	ax.set_xlabel(r"$\eta_x$", fontsize=fsa)
	ax.set_ylabel(r"$\eta_y$", fontsize=fsa)
	ax.set_title(r"$\rho(\eta_x,\eta_y)$ data", fontsize=fsa)
	
	if pred:
		ax = axs[2][1]
		ax.contourf(etax,etay,rhoPexey.T)
	
		ax.set_xlabel(r"$\eta_x$", fontsize=fsa)
		ax.set_title(r"$\rho(\eta_x,\eta_y)$ prediction ($R=S$)", fontsize=fsa)
	
	## ------------------------------------------------------------------------
	
	fig.suptitle(r"PDF projections. $\alpha=%0.1f,R=%0.1f,S=%0.1f$"%(a,R,S), fontsize=fst)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFxy2d"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)

	return
	
	

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
