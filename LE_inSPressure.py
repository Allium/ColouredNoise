
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, optparse
import warnings
from time import time

from LE_Utils import save_data, filename_pars
from LE_inSBS import force_linin

warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in log",
	RuntimeWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in power",
	RuntimeWarning)

## Global variables
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	NAME
		LE_inSPressure.py
		
	PURPOSE
		Plot PDF and pressure against r for files produced by LE_inSBS.py --
		i.e. when wall is a disc at the origin and concentration at infinity fixed.
	
	EXECUTION
		python LE_inSPressure.py [path] [flags]
	
	ARGUMENTS
		histfile	path to density histogram
		dirpath 	path to directory containing histfiles
		
	FLAGS
		-v	--verbose
		-s	--show
			--nosave	False
		-a	--plotall
	"""
	me = "LE_inSPressure.main: "
	t0 = time()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option("-s","--show",
		dest="showfig", default=False, action="store_true")
	parser.add_option("-P","--plotpress",
		dest="plotP", default=False, action="store_true")
	parser.add_option("-v","--verbose",
		dest="verbose", default=False, action="store_true")
	parser.add_option("--logplot","--plotlog",
		dest="logplot", default=False, action="store_true")
	parser.add_option("--nosave",
		dest="nosave", default=False, action="store_true")
	parser.add_option("-a","--plotall",
		dest="plotall", default=False, action="store_true")
	parser.add_option("-h","--help",
		dest="help", default=False, action="store_true")		
	opt, args = parser.parse_args()
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	plotP	= opt.plotP
	verbose = opt.verbose
	logplot	= opt.logplot
	nosave	= opt.nosave
	plotall = opt.plotall
	
	assert "_IN" in args[0], me+"Must be an IN file."
	
	if plotall and os.path.isdir(args[0]):
		showfig = False
		allfiles(args[0],plotP,verbose)
		
	elif os.path.isfile(args[0]):
		pressure_pdf_file(args[0],plotP,verbose)
	elif os.path.isdir(args[0]):
		pressure_dir(args[0],logplot,nosave,verbose)
	else:
		raise IOError, me+"You gave me rubbish. Abort."
	
	if verbose: print me+"execution time",round(time()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_file(histfile, plotpress, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_inSPressure.pressure_pdf_file: "
	t0 = time()

	## Filename
	plotfile = os.path.dirname(histfile)+"/PDF"+os.path.basename(histfile)[4:-4]+".jpg"
	
	## Get pars from filename
	a = filename_par(histfile, "_a")
	S = filename_par(histfile, "_S")
	R,lam,nu = None, None, None
	ftype = "linin"
	if verbose: print me+"alpha =",a,"and S =",S
	
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	erbins = bins["erbins"]
	er = 0.5*(erbins[1:]+erbins[-1])
	rini = rmax	## Start point for computing pressures
	rinid = -1#np.argmin(np.abs(r-rini))
	
	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	try:	H = H.sum(axis=2)
	except ValueError:	pass
	## Noise dimension irrelevant here
	H = np.trapz(H, x=er, axis=1)
	## rho is probability density. H is probability at r
	rho = H/(2*np.pi*r) #/ np.trapz(H, x=r, axis=0)
	rho /= rho[-1]
	
	## White noise result
	r_WN = np.linspace(r[0],r[-1]*(1+0.5/r.size),r.size*5+1)
	rho_WN = pdf_WN(r_WN,[R,S,lam,nu],ftype,verbose)
		
	##---------------------------------------------------------------			
	## PLOT SET-UP
	
	if not plotpress:
		## Only pdf plot
		figtit = "Density; "
		fig, ax = plt.subplots(1,1)
	elif plotpress:
		figtit = "Density and pressure; "
		fig, axs = plt.subplots(2,1,sharex=True)
		ax = axs[0]
		plotfile = os.path.dirname(plotfile)+"/PDFP"+os.path.basename(plotfile)[3:]
	figtit += ftype+"; $\\alpha="+str(a)+"$, $S = "+str(S)+"$"
	# xlim = [S-2*lam,R+2*lam] if (ftype[-3:]=="tan" or ftype[-2:]=="nu") else [S-4.0,R+4.0]
		
	##---------------------------------------------------------------	
	## PDF PLOT
	
	## Wall
	plot_wall(ax, ftype, [R,S,lam,nu], r)
	## PDF and WN PDF
	ax.plot(r,rho,   "b-", label="CN simulation")
	ax.plot(r_WN,rho_WN,"r-", label="WN theory")
	## Accoutrements
	# ax.set_xlim(xlim)
	# ax.set_ylim(bottom=0.0, top=min(20,round(max(rho.max(),rho_WN.max())+0.05,1)))
	ax.set_ylim(bottom=0.0, top=min(20,1.2*max(rho.max(),rho_WN.max())))
	if not plotpress: ax.set_xlabel("$r$", fontsize=fsa)
	ax.set_ylabel("$\\rho(r,\\phi)$", fontsize=fsa)
	ax.grid()
	
	##---------------------------------------------------------------
	## PRESSURE
	
	if plotpress:
	
		## CALCULATIONS
		p	= calc_pressure(r,rho,ftype,[R,S,lam,nu],spatial=True)
		p_WN = calc_pressure(r_WN,rho_WN,ftype,[R,S,lam,nu],spatial=True)
		## Eliminate negative values
		if ftype[0] == "d":
			p		-= p.min()
			p_WN	-= p_WN.min()
		
		##-----------------------------------------------------------
		## PRESSURE PLOT
		ax = axs[1]
		## Wall
		plot_wall(ax, ftype, [R,S,lam,nu], r)
		## Pressure and WN pressure
		ax.plot(r,p,"b-",label="CN simulation")
		ax.plot(r_WN,p_WN,"r-",label="WN theory")
		## Accoutrements
		# ax.set_ylim(bottom=0.0, top=round(max(p.max(),p_WN.max())+0.05,1))
		ax.set_ylim(bottom=0.0, top=min(20,float(1.2*max(p.max(),p_WN.max()))))
		ax.set_xlabel("$r$", fontsize=fsa)
		ax.set_ylabel("$P(r)$", fontsize=fsa)
		ax.grid()
	
	##---------------------------------------------------------------
	
	## Tidy figure
	fig.suptitle(figtit,fontsize=fst)
	fig.tight_layout();	plt.subplots_adjust(top=0.9)	
		
	fig.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
	
	return
	
##=============================================================================
def allfiles(dirpath, plotP, verbose):
	for filepath in np.sort(glob.glob(dirpath+"/BHIS_**.npy")):
		pressure_pdf_file(filepath, plotP, verbose)
		plt.close()
	return
	
##=============================================================================

def calc_pressure(r,rho,ftype,fpars,spatial=False):
	"""
	Calculate pressure given density a a function of coordinate.
	"""
	R, S, lam, nu = fpars
	
	## Calculate force array
	if ftype == "linin":	force = force_linin(r,r,S)
	else: raise ValueError, me+"ftype not recognised."
	
	## Pressure
	if spatial == True:
		P = +np.array([np.trapz(force[i:]*rho[i:], r[i:]) for i in range(1,r.size+1)])
	else:
		raise
	
	return P

##=============================================================================
def pdf_WN(r,fpars,ftype,vb=False):
	"""
	Theoretical radial pdf of a white noise gas.
	"""
	me = "LE_inSPressure.pdf_WN: "
	R, S = fpars[:2]
	Sind = np.argmin(np.abs(r-S))
	if ftype is "linin":
		rho_WN = np.hstack([np.exp(-0.5*(S-r[:Sind])**2),np.ones(r.size-Sind)])
	else: raise IOError
	return rho_WN
	
	
def plot_wall(ax, ftype, fpars, r):
	"""
	Plot the wall profile of type ftype on ax
	"""
	me = "LE_inSPressure.plot_wall: "
	R, S, lam, nu = fpars
	Sidx = np.abs(S-r).argmin()
	## Plot potentials
	if ftype is "linin":
		Ufn = lambda Dr: 0.5*np.power(Dr,2.0)
		ax.plot(r,np.hstack([Ufn(S-r[:Sidx]),np.zeros(r.size-Sidx)]),"k--",label="Potential")
	return
	
	
def filename_par(filename, searchstr):
	"""
	Scrape filename for parameters and return a dict.
	"""
	start = filename.find(searchstr) + len(searchstr)
	finish = start + 1
	while unicode(filename[start:].replace(".",""))[:finish-start].isnumeric():
		finish += 1
	return float(filename[start:finish])
	
	
##=============================================================================
if __name__=="__main__":
	main()
