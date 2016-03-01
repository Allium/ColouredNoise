
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
from itertools import chain

from LE_LightBoundarySim import lookup_xmax,calculate_xmin,calculate_xini,\
		calculate_xbin,calculate_ybin
from LE_Utils import FBW_soft as force_x
from LE_Utils import save_data, filename_pars

warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)

## Global variables
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()


def main():
	"""
	NAME
		LE_Pressure.py
	
	PURPOSE
		Calculate pressure in vicinity of linear potential for particles driven
		by exponentially correlated noise.
	
	EXECUTION
		python LE_Pressure.py histfile/directory flags
	
	ARGUMENTS
		histfile	path to density histogram
		directory 	path to directory containing histfiles
	
	OPTIONS
	
	FLAGS
		-s --show
		-v --verbose
		-a --plotall	Plot each individual file in directory and then do dirplot
		-n --normIG		Divide pressure by corresponding IG result
		-2 --twod		Plot in two dimensions (rather than overlaid 1D plots)
	
	EXAMPLE
		python LE_Pressure.py dat_LE_stream\b=0.01\BHIS_y0.5bi50r5000b0.01X1seed65438.npy
		
	NOTES
	
	BUGS / TODO
		-- 2D PDF doen't work with unequal bin width
		-- D!=0 has been left behind
	
	HISTORY
		12 November 2015	Started
		15 November 2015	Pressure versus alpha functionality
		30 November 2015	Added IG result
		24 February 2016	Merged P_Xa: plotting multiple Xs in 1D or 2D
	"""
	me = "LE_Pressure.main: "
	t0 = sysT()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('-2','--twod',
		dest="twod", default=False, action="store_true")
	parser.add_option('--rawP',
		dest="rawP", default=False, action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")		
	opt = parser.parse_args()[0]
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	plotall = opt.plotall
	twod 	= opt.twod
	normIG	= not opt.rawP
	
	argv[1] = argv[1].replace("\\","/")
	if plotall and os.path.isdir(argv[1]):
		showfig = False
		allfiles(argv[1],verbose)
	if os.path.isfile(argv[1]):
		pressure_pdf_plot_file(argv[1],verbose)
	elif os.path.isdir(argv[1]):
		pressure_plot_dir(argv[1],verbose, twod, normIG)
	else:
		print me+"you gave me rubbish. Abort."
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
	plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".png"
		
	## Get pars from filename
	pars = filename_pars(histfile)
	[alpha,X,D,dt,ymax,R] = [pars[key] for key in ["a","X","D","dt","ymax","R"]]
	assert (R is None), me+"You are using the wrong program. R should not enter."
	if verbose: print me+"alpha =",alpha,"and X =",X,"and D =",D
	
	## Load data
	H = np.load(histfile)
	
	## Space -- for axes
	xmin = calculate_xmin(X,alpha)
	xmax = lookup_xmax(X,alpha)
	xini = calculate_xini(X,alpha)
	try:
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		xbins = bins["xbins"]
		ybins = bins["ybins"]
	except (IOError, KeyError):
		xbins = calculate_xbin(xini,X,xmax,H.shape[1])
		ybins = calculate_ybin(-ymax,ymax,H.shape[0]+1)
	x = 0.5*(xbins[1:]+xbins[:-1])	
	y = 0.5*(ybins[1:]+ybins[:-1])
		
	## Marginalise to PDF in x
	Hx = np.trapz(H,x=y,axis=0)
	Hx /= np.trapz(Hx,x=x)
	
	
	## Calculate pressure
	force = force_x(x,1.0,X,D)
	press = pressure_x(force,Hx,x)
	xIG, forceIG, HxIG, pressIG = ideal_gas(x, X, D, dt)
	
	## PLOTTING
	fig,axs = plt.subplots(1,2)
	
	## Density plot
	ax = axs[0]
	ax.plot(x,Hx,"b-",label="Simulation")
	ax.plot(xIG,HxIG,"r-",label="White noise")
	ax.plot(xIG,-forceIG,"m:",linewidth=2,label="Force")
	ax.set_xlim(left=xini,right=max(xmax,xIG[-1]))
	ax.set_ylim(bottom=0.0,top=1.0/(X-xmin)+0.1)
	ax.set_xlabel("$x$",fontsize=fsa)
	ax.set_ylabel("PDF $\\rho(x)$",fontsize=fsa)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
	
	## Pressure plot
	ax = axs[1]
	ax.plot(x,press,"b-",linewidth=1, label="CN")
	ax.axhline(y=press[-1],color="b",linestyle="--",linewidth=1)
	ax.plot(xIG,pressIG,"r-",label="WN")
	ax.axhline(y=1/(1+X-xini),color="r",linestyle="--",linewidth=1)
	ax.set_xlim(left=xbins[0],right=xbins[-1])
	ax.set_ylim(bottom=0.0)
	ax.set_xlabel("$x$",fontsize=fsa)
	ax.set_ylabel("Pressure",fontsize=fsa)
	ax.grid()
	# ax.legend(loc="best",fontsize=fsl)
	
	plt.tight_layout()
	fig.suptitle("$x_{w}=$"+str(X)+", $\\alpha=$"+str(alpha)+", $\\Delta=$"+str(D),fontsize=16)
	plt.subplots_adjust(top=0.9)
	
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
		
	return fig
	
##=============================================================================
def allfiles(dirpath, verbose):
	for filepath in glob.glob(dirpath+"/*.npy"):
		pressure_pdf_plot_file(filepath, verbose)
		plt.clf(); plt.close()
	return
	
##=============================================================================
##=============================================================================
def pressure_plot_dir(dirpath, verbose, twod=False, normIG=False):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	
	Be careful heed changes in parameters between files in directory
	"""
	me = "LE_Pressure.pressure_plot_dir: "
	t0 = sysT()
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/*.npy"))
	numfiles = len(histfiles)
	if verbose: print me+"found",numfiles,"files"
		
	## ----------------------------------------------------

	## Initialise
	Alpha = np.zeros(numfiles)
	X = np.zeros(numfiles)
	Press = np.zeros(numfiles)
	xmin, xmax = np.zeros([2,numfiles])
	PressIG = np.zeros(numfiles)
		
	## Loop over files
	for i,histfile in enumerate(histfiles):
		
		## Get pars from filename
		pars = filename_pars(histfile)
		[Alpha[i],X[i],D,dt,ymax,R] = [pars[key] for key in ["a","X","D","dt","ymax","R"]]
		assert (R is None), me+"You are using the wrong program. R should not enter."
				
		## Load data
		H = np.load(histfile)
		
		## Space
		xmin[i] = calculate_xmin(X[i],Alpha[i])
		xmax[i] = lookup_xmax(X[i],Alpha[i])
		xini = calculate_xini(X[i],Alpha[i])
		try:
			bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
			xbins = bins["xbins"]
			ybins = bins["ybins"]
		except (IOError, KeyError):
			xbins = calculate_xbin(xini,X[i],xmax[i],H.shape[1])
			ybins = calculate_ybin(0.0,ymax,H.shape[0]+1)
		x = 0.5*(xbins[1:]+xbins[:-1])	
		y = 0.5*(ybins[1:]+ybins[:-1])
		
		## Marginalise to PDF in x and normalise
		Hx = np.trapz(H,x=y,axis=0)
		Hx /= np.trapz(Hx,x=x,axis=0)

		## Calculate pressure
		force = force_x(x,1.0,X[i],D)
		Press[i] = np.trapz(-force*Hx, x)
	
	## ----------------------------------------------------
	## Sort values
	sortind = np.argsort(Alpha)
	Alpha = Alpha[sortind]
	Press = Press[sortind]
	X = X[sortind]
	# xmin = xmin[sortind]
	
	if verbose: print me+"data collection",round(sysT()-t0,2),"seconds."
	
	pressplot = dirpath+"/ALPH_X"+"_dt"+str(dt)+".png"
	
	## ----------------------------------------------------
	## Choose plot type
	
	## 1D
	if twod is False or ((X==X[0]).all() and (Alpha!=Alpha[0]).any()):
	
		XX = np.unique(X)
		Ncurv = XX.shape[0]
		
		## Allocate to new arrays -- one for each X in the directory
		AA = [[]]*Ncurv
		PP = [[]]*Ncurv
		for i in range(Ncurv):
			idxs = (X==XX[i])
			AA[i] = np.array(Alpha[idxs])
			PP[i] = np.array(Press[idxs])
				
		labels = ["X = "+str(XX[i]) for i in range(Ncurv)]
			
		## Calculate IG on finer grid, assuming same X, xmin and dt
		AAIG = AA
		PPIG = [[]]*Ncurv
		if D==0.0:
			PPIG = [1.0/(1.0-np.exp(-4.0)+XX[i]-calculate_xini(XX[i],AA[i])) for i in range(Ncurv)]
		else:
			## Needs update!
			raise AttributeError, me+"no can do."
			PPIG = [ideal_gas(a,x,X,D,dt)[3][-1]/dt for a in AAIG]
			
		if normIG: PP = [PP[i]/PPIG[i] for i in range(Ncurv)]
			
		for i in range(Ncurv):
			plt.plot(AA[i], PP[i], 'o-', label=labels[i])
			# plt.errorbar(AA[i], PP[i], yerr=0.05, color=plt.gca().lines[-1].get_color(), fmt='.', ecolor='grey', capthick=2)
			if not normIG: plt.axhline(PressIG[i], color=plt.gca().lines[-1].get_color(), linestyle="--")
		plt.xlim(right=max(chain.from_iterable(AA)))
		plt.ylim(bottom=0.0)
		plt.title("Pressure normalised by WN result",fontsize=fst)
		plt.xlabel("$\\alpha=(f_0^2\\tau/T\\zeta)^{1/2}$",fontsize=fsa)
		plt.ylabel("Pressure",fontsize=fsa)
		plt.grid()
		plt.legend(loc="best",fontsize=fsl)
	
	## 2D
	else:
		pressplot = pressplot[:-4]+"_2.png"
		
		## IMSHOW
		## Normalise Press by WN value
		Pim = [Press *(1.0+0.1*X) if normIG else Press]
		## Restructure dara, assuming sorted by alpha value
		Aim = np.unique(Alpha)
		
		## Create 2D array of pressures
		## There must be a nicer way of doing this
		Xim = np.unique(X)
		Pim = -np.ones((Xim.shape[0],Aim.shape[0]))
		for k in range(Press.shape[0]):
			for i in range(Xim.shape[0]):
				for j in range(Aim.shape[0]):
					if Aim[j]==Alpha[k] and Xim[i]==X[k]:
						Pim[i,j]=Press[k]
		## Mask zeros
		Pim = np.ma.array(Pim, mask = Pim<0.0)
						
		## Make plot
		im = plt.imshow(Pim[::-1],aspect='auto',extent=[Aim[0],Aim[-1],Xim[0],Xim[-1]],interpolation="none")
		
		# ## CONTOUR
		# ## Messily normalise by WN result
		# Pmesh = Press * (1.0+0.1*X)
		# ## Create coordinate mesh
		# Amesh,Xmesh = np.meshgrid(Alpha,X)
		# Pmesh = np.empty(Xmesh.shape)
		# ## Messily create corresponding pressure mesh
		# mask = []
		# for k in range(Press.shape[0]):
			# for i in range(Xmesh.shape[0]):
				# for j in range(Xmesh.shape[1]):
					# if Amesh[i,j]==Alpha[k] and Xmesh[i,j]==X[k]:
						# Pmesh[i,j]=Press[k]
		# ## Plot using contours
		# plt.contour(Amesh,Xmesh,Pmesh,5)
				
		## ACCOUTREMENTS
		plt.title("Pressure normalised by WN result",fontsize=fst)
		plt.xlabel("$\\alpha=(f_0^2\\tau/T\\zeta)^{1/2}$",fontsize=fsa)
		plt.ylabel("Wall separation",fontsize=fsa)
		plt.grid(None)
		
		cbar = plt.colorbar(im, ticks=[Pim.min(),Pim.mean(),Pim.max()], orientation="vertical")
		cbar.ax.set_yticklabels(["Low", "Mean", "High"],fontsize=fsl)
	
	## --------------------------------------------------------
	
	plt.savefig(pressplot)
	if verbose: print me+"plot saved to",pressplot
	
	return pressplot


##=============================================================================
##=============================================================================
def pressure_x(force,Hx,x):
	"""
	Calculate the pressure given an array of forces and densities at positions x.
	Returns an array of pressure value at every point in x.
	"""
	me = "LE_Pressure.pressure: "
	press = np.array([np.trapz((-force*Hx)[:i], x[:i]) for i,xi in enumerate(x)])
	return press
	
##=============================================================================
def ideal_gas(x, X, D, dt, up=6):
	"""
	Calculate PDF and pressure for ideal gas
	No alpha in new variables 02.02.2016
	"""
	xbinsIG = np.linspace(x[0],X+4.0,up*len(x)+1)
	xIG = 0.5*(xbinsIG[1:]+xbinsIG[:-1])
	forceIG = force_x(xIG,1.0,X,D)
	## Predicted solution
	UIG = np.array([np.trapz((-forceIG)[:i], x=xIG[:i]) for i in range(len(xIG))])
	HxIG = np.exp(-UIG)
	HxIG /= np.trapz(HxIG,x=xIG)
	## Pressure
	pressIG = pressure_x(forceIG,HxIG,xIG)
	return xIG, forceIG, HxIG, pressIG

##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()
