
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import integrate
from matplotlib import pyplot as plt
from sys import argv
import os, glob
import optparse
import warnings
from time import time as sysT

from LE_Utils import save_data, filename_pars
from LE_LightBoundarySim import lookup_xmax, calculate_xbin, calculate_ybin
from LE_2DLBS import force_2D

warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)

def main():
	"""
	NAME
		LE_2DPressure.py
	
	PURPOSE
		Calculate pressure in vicinity of linear potential for particles driven
		by exponentially correlated noise in two dimensions.
	
	EXECUTION
		python LE_2DPressure.py histfile/directory flags
	
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
		Dir plot still to come
	
	HISTORY
		21 February 2016	Adapted from LE_Pressure
	"""
	me = "LE_2DPressure.main: "
	t0 = sysT()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")		
	opt, arg = parser.parse_args()
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	plotall = opt.plotall
	
	argv[1] = argv[1].replace("\\","/")
	if plotall and os.path.isdir(argv[1]):
		showfig = False
		allfiles(argv[1],verbose)
		
	if os.path.isfile(argv[1]):
		pressure_pdf_plot_file(argv[1],verbose)
	elif os.path.isdir(argv[1]):
		pressure_plot_dir(argv[1],verbose)
	else:
		print me+"You gave me rubbish. Abort."
		exit()
	
	if verbose: print me+"execution time",round(sysT()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_plot_file(histfile, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_Pressure.pressure_pdf_plot_file: "
	t0 = sysT()
	
	## Filenames
	plotfile = os.path.dirname(histfile)+"/PRES"+os.path.basename(histfile)[4:-4]+".png"
	
	## Get pars from filename
	pars = filename_pars(histfile)
	[alpha,X,D,dt,ymax,R] = [pars[key] for key in ["a","X","D","dt","ymax","R"]]
	assert (R is not None), me+"You are using the wrong program. R should be defined."
	assert (D == 0.0), me+"Cannot yet handle soft potential. D should be 0.0."
	if verbose: print me+"alpha =",alpha,"and X =",X,"and D =",D
	
	## Load data and normalise
	H = np.load(histfile)
	H /= H.sum()
		
	## Centre of circle for curved boundary
	c = [X-np.sqrt(R*R-ymax*ymax),0.0]
	## Space (for axes)
	xini, xmax = 0.9*X, lookup_xmax(c[0]+R,alpha)
	ybins = calculate_ybin(0.0,ymax,H.shape[0]+1)
	y = 0.5*(ybins[1:]+ybins[:-1])
	xbins = calculate_xbin(xini,X,xmax,H.shape[1])
	x = 0.5*(xbins[1:]+xbins[:-1])
		
	
	## Set up plot
	fig,axs = plt.subplots(1,2)
	fs = 14
		
	## pdf plot
	ax = axs[0]
	Xm,Ym = np.meshgrid(x,y)
	CS = ax.contourf(Xm,Ym[::-1],H,10)
	cbar = fig.colorbar(CS, ax=ax, orientation="horizontal", ticks=[H.min(),H.mean(),H.max()])
	cbar.ax.set_xticklabels(["Low", "Mean", "High"])
	### http://stackoverflow.com/questions/13310594/positioning-the-colorbar
	## Plot wall
	wallx = np.linspace(X,c[0]+R,201)
	wally = c[1]+np.sqrt(R*R-(wallx-c[0])**2)
	ax.plot(wallx,wally, "r--",linewidth=2)
	## Accoutrements
	ax.set_xlim([xini,xmax])
	ax.set_ylim([0.0,ymax])
	ax.set_xlabel("$x$", fontsize=fs)
	ax.set_ylabel("$y$", fontsize=fs)
		

	## Calculate force array (2d)
	force = -1.0 * ( (Xm-c[0])**2 + (Ym-c[1])**2 > R*R ) * ( Xm-c[0]>0.0 )
	## Pressure array (2d) -- sum rather than trapz
	press = -1.0*(force*H).sum(axis=0).cumsum(axis=0)
	
	## Pressure plot
	ax = axs[1]
	ax.plot(x,press,label="CN simulation")
	## Bulk and wall regions
	plt.axvspan(xini,X, color="b",alpha=0.1) 
	plt.axvspan(X,c[0]+R, color="m",alpha=0.05)
	plt.axvspan(R,xmax, color="r",alpha=0.05)
	## Ideal gas result
	ax.hlines(pressIG(ymax,R,c[0]),xini,xmax,linestyle="-",color="g",label="WN theory")
	## Accoutrements
	ax.set_xlim([xini,xmax])
	ax.set_xlabel("$x$", fontsize=fs)
	ax.set_ylabel("Pressure, $P$", fontsize=fs)
	ax.grid()
	ax.legend(loc="best",fontsize=11)
	
	
	## Tidy figure
	fig.suptitle(os.path.basename(plotfile))
	fig.tight_layout()
	plt.subplots_adjust(top=0.9)	
		
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
		
	return fig
	
##=============================================================================
def allfiles(dirpath, verbose):
	for filepath in glob.glob(dirpath+"/BHIS_2D_*.npy"):
		pressure_pdf_plot_file(filepath, verbose)
		plt.clf()
	return

##=============================================================================
def pressure_plot_dir(dirpath, verbose):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	
	Be careful heed changes in parameters between files in directory
	"""
	me = "LE_Pressure.pressure_plot_dir: "
	t0 = sysT()
	return
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/BHIS_2D_*.npy"))
	numfiles = len(histfiles)
	if verbose: print me+"found",numfiles,"files"
	
	## Outfile name
	pressplot = dirpath+"/PressureAlpha.png"
	Alpha = np.zeros(numfiles)
	Press = np.zeros(numfiles)
	PressIG = np.zeros(numfiles)
		
	## Loop over files
	for i,filepath in enumerate(histfiles):
		
		## Find alpha
		Alpha[i], X, D, dt, ymax = filename_pars(filepath)
				
		## Load data
		H = np.load(filepath)
		
		## Space
		xmin,xmax = 0.9*X,lookup_xmax(X,Alpha[i])
		ymax = 0.5
		x = calculate_xbin(xmin,X,xmax,H.shape[1]-1)
		y = np.linspace(-ymax,ymax,H.shape[0])
		
		## Marginalise to PDF in X
		Hx = np.trapz(H,x=y,axis=0)
		Hx /= np.trapz(Hx,x=x,axis=0)

		## Calculate pressure
		force = force_x(x,1.0,X,D)
		Press[i] = np.trapz(-force*Hx, x)
	
	## Sort values
	sortind = np.argsort(Alpha)
	Alpha = Alpha[sortind]; Press = Press[sortind]; PressIG = PressIG[sortind]
	
	## Calculate IG pressure on a finer grid -- assume X, dt same
	tIG = sysT()
	AlphaIG = Alpha
	if D==0.0:
		PressIG = 1.0/(1.0+X-xmin) * np.ones(len(AlphaIG))
	else:
		PressIG = [ideal_gas(a,x,X,D,dt)[3][-1]/dt for a in AlphaIG]
	if verbose: print me+"white noise pressure calculation:",round(sysT()-tIG,2),"seconds."
		
	## Plotting
	plt.errorbar(Alpha, Press, yerr=0.05, fmt='bo', ecolor='grey', capthick=2,label="Simulated")
	# plt.plot(AlphaIG,PressIG,"r-",label="White noise")
	plt.axhline(PressIG[0], color="r",linestyle="-",label="White noise")
	plt.ylim(bottom=0.0)
	plot_acco(plt.gca(), xlabel="$\\alpha=f_0^2\\tau/T\\zeta$", ylabel="Pressure")
	
	plt.savefig(pressplot)
	if verbose: print me+"plot saved to",pressplot
	
	return pressplot


##=============================================================================
	
##=============================================================================
def pressIG(ym,R,cx):
	"""
	Theoretical pressure of a white noise gas.
	See notes 22/02/2016
	"""
	return 1.0/(2*ym*(1+cx)+ym*np.sqrt(R*R-ym*ym)+R*R*np.arcsin(ym/R))


##=============================================================================
if __name__=="__main__":
	main()
