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
from LE_Utils import fs, set_mplrc
from LE_SPressure import plot_wall

set_mplrc(fs)

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
	vb = opt.verbose
		
	if os.path.isfile(args[0]):
		plot_pdf1d(args[0], nosave, vb)
		plot_pdf2d(args[0], nosave, vb)
	elif (plotall and os.path.isdir(args[0])):
		showfig = False
		filelist = np.sort(glob.glob(args[0]+"/BHIS_*"+searchstr+"*.npy"))
		if vb: print me+"Found",len(filelist),"files."
		for histfile in filelist:
			plot_pdf1d(histfile, nosave, vb)
			plot_pdf2d(histfile, nosave, vb)
			plt.close()
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
	
	##-------------------------------------------------------------------------
	
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
	## Fit
	gauss = lambda x, m, s2:\
				1/(2*np.pi*(s2*np.exp(-0.5*m**2/s2)+\
				+np.abs(m)*np.sqrt(np.pi*s2/2)*(1+sp.special.erf(np.abs(m)/(np.sqrt(2*s2))))))*\
				np.exp(-0.5*(x-m)**2/s2)
	
	if R==S: fitQ = sp.optimize.curve_fit(gauss, r, Q, p0=[R,1/np.sqrt(1+a)])[0]
	fitq = sp.optimize.curve_fit(gauss, eta, q, p0=[0,1/a])[0]
	
	##-------------------------------------------------------------------------
	
	fig, axs = plt.subplots(2,1)
	fig.canvas.set_window_title("1D PDFs")
	
	## Spatial density plot
	ax = axs[0]
	
	## Data
	ax.plot(r, Q, label=r"Simulation")
	
	## Gaussian
	if R==S:
		ax.plot(r, gauss(r,fitQ[0],1/(1+a)), label=r"$G\left(\mu, \frac{1}{\alpha+1}\right)$")
	
	## Potential
	if "_DL_" in histfile:
		ax.plot(r, (r-R)**2 * Q.max()/((r-R)**2).max(), "k--", label=r"$U(r)$")
	
	ax.set_xlabel(r"$r$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$Q(r)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="upper right", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
		
	##-------------------------------------------------------------------------
	
	## Force density plot
	ax = axs[1]
	
	## Data
	ax.plot(eta, q, label=r"Simulation")
	
	## Gaussian
	ax.plot(eta, gauss(eta,0,1/a), label=r"$G\left(0, \frac{1}{\alpha}\right)$")
	
	ax.set_xlabel(r"$\eta$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$q(\eta)$", fontsize=fs["fsa"])
	ax.grid()
	ax.legend(loc="upper right", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	
	##-------------------------------------------------------------------------
	
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	fig.suptitle(r"PDFs in $r$ and $\eta$. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S), fontsize=fs["fst"])
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFre1d"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
	
	return
	
	
##=============================================================================
def plot_pdf2d(histfile, nosave, vb):
	"""
	Read in data for a single file and plot 3D PDF.
	"""
	me = me0+".plot_pdf2D: "
	t0 = time.time()

	##-------------------------------------------------------------------------
	
	## Get pars from filename
	
	assert ("_POL_" in histfile or "_CIR_" in histfile), me+"Functional only for Cartesian geometry."
	
	a = filename_par(histfile, "_a")
	R = filename_par(histfile, "_R")
	S = filename_par(histfile, "_S")
	
	##-------------------------------------------------------------------------
	
	## Space
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	ebins = bins["erbins"]
	pbins = bins["epbins"]
	r = 0.5*(rbins[1:]+rbins[:-1])
	eta = 0.5*(ebins[1:]+ebins[:-1])
	psi = 0.5*(pbins[1:]+pbins[:-1])
	
	## Wall indices
	Rind, Sind = np.abs(r-R).argmin(), np.abs(r-S).argmin()
		
	##-------------------------------------------------------------------------
	## Histogram / density
	
	H = np.load(histfile)
	rho =  H / (H.sum() * (r[1]-r[0])*(eta[1]-eta[0])*(psi[1]-psi[0]))
			
	## ------------------------------------------------------------------------
	## Projections
	
	rhore = rho.sum(axis=2)
	rhorp = rho.sum(axis=1)
	rhoep = rho.sum(axis=0)
	
	## ------------------------------------------------------------------------
	## Plotting
	
	fig, axs = plt.subplots(3,1, sharey=True, figsize=(10,10))
	fig.canvas.set_window_title("2D PDFs")
	
	plt.rcParams["image.cmap"] = "Greys"#"coolwarm"
	
	## ------------------------------------------------------------------------
	
	## r-eta
	
	ax = axs[0]
	ax.contourf(r,eta,rhore.T)
	ax.axvline(R,c="k"); ax.axvline(S,c="k")
	
	ax.set_xlabel(r"$r$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$\eta$", fontsize=fs["fsa"])
	ax.set_title(r"$\rho(r,\eta)$ data", fontsize=fs["fsa"])
	
	## r-psi
	
	ax = axs[1]
	ax.contourf(r,psi,rhorp.T)
	ax.axvline(R,c="k"); ax.axvline(S,c="k")
	
	ax.set_xlabel(r"$r$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$\psi$", fontsize=fs["fsa"])
	ax.set_title(r"$\rho(r,\psi)$ data", fontsize=fs["fsa"])
	
	## etax-etay
	
	ax = axs[2]
	ax.contourf(eta,psi,rhoep.T)
	
	ax.set_xlabel(r"$\eta$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$\psi$", fontsize=fs["fsa"])
	ax.set_title(r"$\rho(\eta,\psi)$ data", fontsize=fs["fsa"])
	
	## ------------------------------------------------------------------------
	
	title = r"PDF projections. $\alpha=%.1f, R=%.1f, S=%.1f$"%(a,R,S)
	fig.suptitle(title, fontsize=fs["fst"])
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	
	## ------------------------------------------------------------------------
	
	if not nosave:
		plotfile = os.path.dirname(histfile)+"/PDFre2d"+os.path.basename(histfile)[4:-4]+".jpg"
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
		
	if vb: print me+"Execution time %.1f seconds."%(time.time()-t0)

	return
	
##=============================================================================
def plot_fitpars(histdir, searchstr, nosave, vb):
	"""
	For each file in directory, plot offset of Q(r) peak from R.
	"""
	me = me0+".plot_fitpars: "
	
	##-------------------------------------------------------------------------
	## Fit function
	gauss = lambda x, m, s2:\
				1/(2*np.pi*(s2*np.exp(-0.5*m**2/s2)+\
				+m*np.sqrt(np.pi*s2/2)*(1+sp.special.erf(m/(np.sqrt(2*s2))))))*\
				np.exp(-0.5*(x-m)**2/s2)
	##-------------------------------------------------------------------------
	
	filelist = np.sort(glob.glob(histdir+"/BHIS_*"+searchstr+"*.npy"))
	numfiles = filelist.size
	assert numfiles>1, me+"Check directory."
	if vb: print me+"Found",numfiles,"files."
	
	A, M, S2 = np.zeros((3,numfiles))
	
	for i, histfile in enumerate(filelist):
	
		## Get pars from filename
		A[i] = filename_par(histfile, "_a")
		R = filename_par(histfile, "_R")
		S = filename_par(histfile, "_S")
		assert R==S, me+"Functionality only for zero bulk."
		
		## Space
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		rbins = bins["rbins"]
		ebins = bins["erbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		eta = 0.5*(ebins[1:]+ebins[:-1])
	
		## Histogram
		H = np.load(histfile)
		try: H = H.sum(axis=2)
		except ValueError: pass
	
		## Spatial density
		Q = H.sum(axis=1)*(eta[1]-eta[0]) / (2*np.pi*r) / np.trapz(np.trapz(H,eta,axis=1),r,axis=0)
		M[i], S2[i] = sp.optimize.curve_fit(gauss, r, Q, p0=[R,1/(1+A[i])])[0]
	
	srtidx = A.argsort()
	A = A[srtidx]
	M = M[srtidx]
	S2 = S2[srtidx]
	S1 = np.sqrt(S2)
		
	##-------------------------------------------------------------------------
	## Fit
	
	linear = lambda x, m, c: m*x + c
	
	fitS = sp.optimize.curve_fit(linear, np.log(1+A), np.log(S1), p0=[-0.5,0.0])[0]
	
	##-------------------------------------------------------------------------
	
	fig, ax = plt.subplots(1,1)
	
	## Plot data -- sigma
	lineS = ax.plot(1+A, S1, "o", label=r"$\sigma$ (data)")
	## Plot fit
	ax.plot(1+A, np.exp(linear(np.log(1+A), *fitS)), lineS[0].get_color()+"--",\
									label=r"$%.1g(1+\alpha)^{%.2g}$"%(np.exp(fitS[1]),fitS[0]))
	## Plot prediction
#	ax.plot(1+A, np.exp(linear(np.log(1+A), -0.5, 0.0)), lineS[0].get_color()+":", label=r"$(1+\alpha)^{-1/2}$")
	
	## Plot data -- mean
	lineM = ax.plot(1+A, M-R, "o-", label=r"$\mu-R$ (data)")
		
	ax.set_xlim(1.0, 1.0+A[-1])
	ax.set_xscale("log")
	ax.set_yscale("log")
		
	ax.set_xlabel(r"$1+\alpha$", fontsize=fs["fsa"])
	ax.set_ylabel(r"Parameter", fontsize=fs["fsa"])
	ax.set_title(r"Gaussian fit parameters. $R=%.1f, S=%.1f$"%(R,S), fontsize=fs["fst"])
	ax.grid()
	ax.legend(loc="best", fontsize=fs["fsl"]).get_frame().set_alpha(0.5)
	
	if not nosave:
		plotfile = histdir+"/PDFparsa_R%.1f_S%.1f.jpg"%(R,S)
		fig.savefig(plotfile)
		fig.savefig(plotfile)
		if vb:	print me+"Figure saved to",plotfile
	
	##-------------------------------------------------------------------------
	
	return

##=============================================================================
##=============================================================================
if __name__ == "__main__":
	main()
