
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os, glob, optparse
import warnings
from time import time

from LE_Utils import save_data, filename_pars


warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in sign",
	RuntimeWarning)

## Global variables
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	NAME
		LE_SPressure.py
	
	PURPOSE
	
	EXECUTION
	
	ARGUMENTS
		histfile	path to density histogram
		directory 	path to directory containing histfiles
	
	OPTIONS
	
	FLAGS
		-v --verbose
		-s --show
		-a --plotall
	
	EXAMPLE
		
	NOTES
	
	BUGS / TODO
	
	HISTORY
		21/03/2016	Started
	"""
	me = "LE_SPressure.main: "
	t0 = time()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('--rawp',
		dest="rawp", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('-h','--help',
		dest="help", default=False, action="store_true")		
	opt, args = parser.parse_args()
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	rawp	= opt.rawp
	plotall = opt.plotall
	
	# args[0] = args[0].replace("\\","/")
	if plotall and os.path.isdir(args[0]):
		showfig = False
		allfiles(args[0],verbose)
		
	if os.path.isfile(args[0]):
		pressure_pdf_file(args[0],verbose)
	elif os.path.isdir(args[0]):
		pressure_dir(args[0],rawp,verbose)
	else:
		raise IOError, me+"You gave me rubbish. Abort."
	
	if verbose: print me+"execution time",round(time()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_file(histfile, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_SPressure.pressure_pdf_file: "
	t0 = time()
	
	## Filenames
	plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".png"
	
	## Get pars from filename
	pars = filename_pars(histfile)
	[a,R] = [pars[key] for key in ["a","R"]]
	assert (R is not None), me+"You are using the wrong program. R must be defined."
	if verbose: print me+"alpha =",a,"and R =",R
		
	## Space (for axes)
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	rbins = bins["rbins"]
	rini = rbins[0]
	rmax = rbins[-1]
	r = 0.5*(rbins[1:]+rbins[:-1])
	
	## Load histogram, convert to normalised pdf
	H = np.load(histfile)
	## Normalise as if extended to r=0
	H = Hr_norm(H,r,R)

	## White noise result
	rho_WN = pdf_WN(r,R)
	
	## Set up plot
	fig,axs = plt.subplots(2,1,sharex=True)
		
	## pdf plot
	ax = axs[0]
	
	ax.plot(r,H, label="CN simulation")
	ax.plot(r,rho_WN, label="WN theory")
	ax.axvline(R,c="k")
	ax.set_xlim(right=rmax)
	ax.set_ylim(bottom=0.0)
	ax.set_ylabel("$\\rho(r)$", fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right",fontsize=fsl)
		
	## Calculate force array
	force = 0.5*(np.sign(R-r)-1)
	## Pressure array -- sum rather than trapz
	p = -(force*H).cumsum() * (r[1]-r[0])
	p_WN = -(force*rho_WN).cumsum() * (r[1]-r[0])
	
	## Pressure plot
	ax = axs[1]
	ax.plot(r,p,label="CN simulation")
	ax.plot(r,p_WN,label="WN theory")
	ax.axvline(R,c="k")
	## Accoutrements
	ax.set_xlim(right=rmax)
	ax.set_xlabel("$r$", fontsize=fsa)
	ax.set_ylabel("$P(r)$", fontsize=fsa)
	ax.grid()
	# ax.legend(loc="lower right",fontsize=fsl)
	
	## Tidy figure
	fig.suptitle(os.path.basename(plotfile),fontsize=fst)
	fig.tight_layout()
	plt.subplots_adjust(top=0.9)	
		
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
	
	return
	
##=============================================================================
def allfiles(dirpath, verbose):
	for filepath in glob.glob(dirpath+"/BHIS_CIR_*.npy"):
		pressure_pdf_file(filepath, verbose)
		plt.close()
	return

##=============================================================================
def pressure_dir(dirpath, rawp, verbose):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	"""
	me = "LE_SPressure.pressure_dir: "
	t0 = time()
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/BHIS_CIR_*.npy"))
	numfiles = len(histfiles)
	if verbose: print me+"found",numfiles,"files"
	
	## Initialise
	A = np.zeros(numfiles) 
	R = np.zeros(numfiles)
	P = np.zeros(numfiles)
	P_WN = np.zeros(numfiles)
	rini = np.zeros(numfiles)
		
	## Loop over files
	for i,histfile in enumerate(histfiles):
	
		## Get pars from filename
		pars = filename_pars(histfile)
		[A[i],R[i]] = [pars[key] for key in ["a","R"]]

		## Space (for axes)
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		rbins = bins["rbins"]
		rini = rbins[0]
		rmax = rbins[-1]
		r = 0.5*(rbins[1:]+rbins[:-1])
		
		## Load (density) histogram, convert to normalised pdf
		H = np.load(histfile)
		H = Hr_norm(H,r,R[i])

		## Calculate force array
		force = 0.5*(np.sign(R[i]-r)-1)
		## Pressure array -- sum rather than trapz
		P[i] = -(force*H).sum() * (r[1]-r[0])
		P_WN[i] = -(force*pdf_WN(r,R[i])).sum() * (r[1]-r[0])
		
	
	## ------------------------------------------------	
	## Create 2D pressure array and 1D a,R coordinate arrays

	## Ordered independent variable arrays
	AA = np.unique(A)
	RR = np.unique(R)
	
	## 3D pressure array: [R,A]
	PP = np.zeros([RR.size,AA.size])
	PP_WN = np.zeros(PP.shape)
	for i in range(RR.size):
		Ridx = (R==RR[i])
		for j in range(AA.size):
			Aidx = (A==AA[j])
			Pidx = Ridx*Aidx
			try:
				PP[i,j] = P[Pidx]
				PP_WN[i,j] = P_WN[Pidx]
			except ValueError:
				## No value there
				pass
	
	## Mask zeros
	PP = np.ma.array(PP, mask = PP==0.0)
	PP_WN = np.ma.array(PP_WN, mask = PP==0.0)
	
	
	## ------------------------------------------------
	## PLOTS
		
	fig, ax = plt.subplots(1,1)
	
	## How to include WN result
	if rawp:
		[ax.plot(AA,PP_WN[i,:],"--",) for i in range(RR.size)]
		ax.set_color_cycle(None)
		title = "Pressure"
		plotfile = dirpath+"/PAR1_rawp.png"
	else:
		PP /= PP_WN
		title = "Pressure normalised by WN"
		plotfile = dirpath+"/PAR1.png"

	for i in range(RR.size):
		ax.plot(AA,PP[i,:],  "o-", label="$R = "+str(RR[i])+"$") 

	ax.set_xlim((AA[0],AA[-1]))
	ax.set_ylim(bottom=0.0)
	ax.set_xlabel("$\\alpha$",fontsize=fsa)
	
	ax.set_ylabel("Pressure",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
	ax.set_title(title)
	plt.tight_layout()
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
		
	return

##=============================================================================
def pdf_WN(r,R):
	"""
	Theoretical radial pdf of a white noise gas.
	"""
	Rind = np.argmin(np.abs(r-R))
	# rho0 = 1.0/(0.5*R*R+R+1.0)
	# rho_WN = rho0 * r * np.hstack([np.ones(Rind),np.exp(R-r[Rind:])])
	rho0 = 1.0/(R+1.0)
	rho_WN = rho0 * np.hstack([np.ones(Rind),np.exp(R-r[Rind:])])
	return rho_WN

def Hr_norm(H,r,R):
	"""
	H is probability density per unit area (flat in the bulk).
	Hr is probability density.
	"""
	H[0]=H[1]
	rext = np.hstack([np.linspace(0.0,r[0],2),r])
	Hext = np.hstack([H[:np.argmin(np.abs(r-R))].mean()*np.ones(2),H])
	# H /= np.trapz(rext*Hext,x=rext)
	H /= np.trapz(Hext,x=rext)
	return H

##=============================================================================
if __name__=="__main__":
	main()
