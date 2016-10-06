
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
import os, glob
import optparse
import warnings
from time import time as sysT
from LE_LightBoundarySim import lookup_xmax, calculate_xbin
from LE_Utils import FBW_soft as force_x
from LE_Utils import save_data
from LE_Pressure import pressure_x, ideal_gas, filename_pars, plot_acco

warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)

def main():
	"""
	NAME
		LE_DeltaPressure.py
	
	PURPOSE
		Plot a graph of pressure against alpha for coloured and white noise for multiple
		values of the potential sofness parameter Delta
		
	STARTED
		13 December 2015
	"""
	me = "LE_DeltaPressure.main: "
	t0 = sysT()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")		
	opt = parser.parse_args()[0]
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	
	## Find relevant directories
	rootdir = argv[1].replace("\\","/")
	filelist = glob.glob(rootdir+"/*.npy")
	dirlist = np.unique([os.path.dirname(histfile) for histfile in filelist])
	ndir = len(dirlist)
	if verbose: print me+"searching in directories "+str(dirlist)+". Found "+str(len(filelist))+" files."
	
	## Initialise arrays
	Alpha_D = []
	Press_D = []
	AlphaIG_D = []
	PressIG_D = []
	D = []
	
	tread = sysT()
	## Calculate pressure as function of alpha for each D
	for i, dirpath in enumerate(dirlist):
		data = pressure_of_alpha(dirpath,verbose)
		Alpha_D += [data[0]]
		Press_D += [data[1]]
		AlphaIG_D += [data[2]]
		PressIG_D += [data[3]]
		D += [data[4]]
	if verbose: print me+"data extraction",round(sysT()-tread,2),"seconds."
	
	sortind = list(np.argsort(D))
	D = np.array(D)[sortind]
	Alpha_D = np.array(Alpha_D)[sortind];		Press_D = np.array(Press_D)[sortind]
	AlphaIG_D = np.array(AlphaIG_D)[sortind];	PressIG_D = np.array(PressIG_D)[sortind]
	
	## Use IG D=0 result as reference
	## Should be the same as running again with alpha in different place -- ???
	## See notes 2015.12.15
	# X,xmin,dt = 10.0,9.0,0.01
	# P_ref = 1/(X-xmin+dt/Alpha_D)
	# PIG_ref = 1/(X-xmin+dt/AlphaIG_D)
	# Press_D /= P_ref
	# PressIG_D /= PIG_ref
	
	tplot = sysT()
	## Plot
	colour = ["b","r","g","m","k"]
	
	for i in range(ndir):
		plt.errorbar(Alpha_D[i], Press_D[i], yerr=0.05,\
			fmt=colour[i]+"o-", ecolor='grey', capthick=2, label="CN; $\\Delta = "+str(D[i])+"$")
		plt.plot(AlphaIG_D[i],PressIG_D[i], colour[i]+"--", label="WN; $\\Delta = "+str(D[i])+"$")
		## Fit
		fitarr, m = make_fit(Alpha_D[i],Press_D[i])
		plt.plot(Alpha_D[i],fitarr,colour[i]+":",label="$P \\sim "+str(m)+"\\alpha$")
	
	plot_acco(plt.gca(), xlabel="$\\alpha=f_0^2\\tau/T\\zeta$", ylabel="Pressure",\
			legloc="")
	
	## Reorder legend
	plot_legend(plt.gca(), ndir)
	
	## Save to each directory
	for dir in dirlist:
		plotfile = dir+"/PressureAlphaDelta.png"
		plt.savefig(plotfile)
		if verbose: print me+"plot saved to "+plotfile
		
	if verbose: print me+"plotting and saving",round(sysT()-tplot,2),"seconds."
	
	if verbose: print me+"total execution time",round(sysT()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
		
##=============================================================================
def pressure_of_alpha(dirpath, verbose):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	
	Be careful heed changes in parameters between files in directory
	"""
	me = "LE_DeltaPressure.pressure_of_alpha: "
	t0 = sysT()
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/*.npy"))
	numfiles = len(histfiles)
	
	Alpha = np.zeros(numfiles)
	Press = np.zeros(numfiles)
	PressIG = np.zeros(numfiles)
		
	## Loop over files
	for i,filepath in enumerate(histfiles):
		
		## Find alpha; assume all other pars stay the same
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
		force = force_x(x,Alpha[i],X,0.01)
		Press[i] = np.trapz(-force*Hx, x)/dt
	
	## Sort values
	sortind = np.argsort(Alpha)
	Alpha = Alpha[sortind]; Press = Press[sortind]; PressIG = PressIG[sortind]
	
	if verbose: print me+" pressure caclulation ",str(round(sysT()-t0,2)),"seconds."
	
	AlphaIG, PressIG = pressureIG_of_alpha(Alpha, Press, x,X,D,dt, verbose)
		
	return [Alpha, Press, AlphaIG, PressIG, D]

##=============================================================================
def pressureIG_of_alpha(Alpha, Press, x,X,D,dt, verbose):
	"""
	Calculate IG pressure on a finer grid
	"""
	me = "LE_DeltaPressure.pressureIG_of_alpha: "
	tIG = sysT()
	AlphaIG = Alpha
	PressIG = [ideal_gas(a,x,X,D,dt,2)[3][-1]/dt for a in AlphaIG]
	if verbose: print me+"white noise pressure calculation:",round(sysT()-tIG,2),"seconds."
	return AlphaIG, PressIG
		
##=============================================================================
def make_fit(x,y):
	initind = np.argmin(np.abs(x-0.1))
	linfit = np.polyfit(x[initind:], y[initind:], 1)
	linfit_fn = np.poly1d(linfit)
	return linfit_fn(x), round(linfit[0],1)
	
##=============================================================================
def plot_legend(ax, ndir):
	sortind = []; sortind += [[2*ndir+i,2*i+1,2*i] for i in range(ndir)]
	sortind = np.array(sortind).flatten()
	handles, labels = ax.get_legend_handles_labels()
	handles, labels = np.array(handles)[sortind], np.array(labels)[sortind]
	ax.legend(handles, labels, loc="best", fontsize=11)
	return

##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()