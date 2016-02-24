
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

from LE_LightBoundarySim import lookup_xmax, calculate_xbin
from LE_Utils import FBW_soft as force_x
from LE_Utils import save_data

warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)

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
	parser.add_option('-n','--normIG',
		dest="normIG", default=False, action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")		
	opt = parser.parse_args()[0]
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	plotall = opt.plotall
	twod 	= opt.twod
	normIG	= opt.normIG
	
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
def pressure_pdf_plot_file(filepath, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_Pressure.pressure_pdf_plot_file: "
	t0 = sysT()
	
	## Filenames
	plotfile = os.path.splitext(filepath)[0]+"_P.png"
	plotfilePDF = os.path.splitext(filepath)[0]+"_PDF.png"
	
	## Get alpha and X from filename
	alpha, X, D, dt, ymax = filename_pars(filepath)
	if verbose: print me+"alpha =",alpha,"and X =",X,"and D =",D
	
	## Load data
	H = np.load(filepath)
	
	## Space -- for axes
	xmin, xmax = 0.9*X, lookup_xmax(X,alpha)
	ybins = np.linspace(-ymax,ymax,H.shape[0]+1)
	y = 0.5*(ybins[1:]+ybins[:-1])
	xbins = calculate_xbin(xmin,X,xmax,H.shape[1])
	x = 0.5*(xbins[1:]+xbins[:-1])
	
	## 2D PDF plot. Cannot do unequal bin widths.
	if 0:
		plt.imshow(H, extent=[xmin,xmax,-ymax,ymax], aspect="auto")
		plt.plot([X,X],[-ymax,ymax],"m:",linewidth=2)
		plt.xlim(xmin,xmax); plt.ylim(-ymax,ymax)
		plot_acco(plt.gca(),xlabel="$x$",ylabel="$\\eta$",title="$\\alpha=$"+str(alpha))
		plt.savefig(plotfilePDF)
		if verbose: print me+"2D PDF plot saved to",plotfilePDF
	## Plot y-pdfs in the bulk.
	if 0:
		plt.clf()
		for i in range(0,H.shape[1]/2,H.shape[1]/20):
			plt.plot(y,H[:,i]/np.trapz(H[:,i],x=y))
		var = 0.05
		plt.plot(y,1/np.sqrt(2*np.pi*var)*np.exp(-y*y/(2*var)),"k--",linewidth=3)
		plot_acco(plt.gca(),xlabel="$\\eta$",ylabel="$p(\\eta)$",\
					title="$\\alpha=$"+str(alpha)+". PDF slices in bulk.")
		plt.savefig(os.path.splitext(filepath)[0]+"_yPDFs.png")
	
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
	# ax.axhline(y=0.0,color="b",linestyle="-",linewidth=1)
	ax.plot(xIG,HxIG,"r--",label="White noise")
	ax.plot(xIG,-forceIG,"m:",linewidth=2,label="Force")
	ax.set_xlim(left=xmin,right=max(xmax,xIG[-1]))
	ax.set_ylim(bottom=0.0,top=1.0/(X-xmin)+0.1)
	plot_acco(ax,ylabel="PDF p(x)",legloc="best")
	
	## Pressure plot
	ax = axs[1]
	ax.plot(x,press,"b-",label="",linewidth=1)
	ax.axhline(y=press[-1],color="b",linestyle="--",linewidth=1)
	ax.plot(xIG,pressIG,"r--",label="")
	ax.axhline(y=1/(1+X-xmin),color="g",linestyle="--",linewidth=1)
	# ax.set_xlim(left=xmin,right=max(xmax,xIG[-1]))
	ax.set_ylim(bottom=0.0)
	
	plot_acco(ax, ylabel="Pressure", legloc="")
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
	
	## Outfile name
	pressplot = dirpath+"/PressureAlphaX.png"
		
	## ----------------------------------------------------

	## Initialise
	Alpha = np.zeros(numfiles)
	X = np.zeros(numfiles)
	Press = np.zeros(numfiles)
	xmin, xmax = np.zeros([2,numfiles])
	PressIG = np.zeros(numfiles)
		
	## Loop over files
	for i,filepath in enumerate(histfiles):
		
		## Find pars; assume D and dt and ymax fixed
		Alpha[i], X[i], D,dt,ymax = filename_pars(filepath)
				
		## Load data
		H = np.load(filepath)
		
		## Space
		xmin[i],xmax[i] = 0.9*X[i],lookup_xmax(X[i],Alpha[i])
		x = calculate_xbin(xmin[i],X[i],xmax[i],H.shape[1]-1)
		y = np.linspace(-ymax,ymax,H.shape[0])
		
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
			AA[i] = Alpha[idxs]
			PP[i] = Press[idxs]
				
		labels = ["X = "+str(XX[i]) for i in range(Ncurv)]
			
		## Calculate IG on finer grid, assuming same X, xmin and dt
		AAIG = AA
		if D==0.0:
			PPIG = 1.0/(1.0+XX-0.9*XX)
		else:
			## Needs update!
			PPIG = [ideal_gas(a,x,X,D,dt)[3][-1]/dt for a in AAIG]

		if normIG: PP = [PP[i]/PPIG[i] for i in range(Ncurv)]
			
		for i in range(Ncurv):
			plt.plot(AA[i], PP[i], 'o-', label=labels[i])
			# plt.errorbar(AA[i], PP[i], yerr=0.05, color=plt.gca().lines[-1].get_color(), fmt='.', ecolor='grey', capthick=2)
			if not normIG: plt.axhline(PressIG[i], color=plt.gca().lines[-1].get_color(), linestyle="--")
		plt.xlim(right=max(chain.from_iterable(AA)))
		plt.ylim(bottom=0.0)
		plot_acco(plt.gca(), xlabel="$\\alpha=(f_0^2\\tau/T\\zeta)^{1/2}$", ylabel="Pressure")
	
	## 2D
	else:
		pressplot = pressplot[:-4]+"2D.png"
		
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
		Pim[Pim<0.0]=np.nan
						
		## Make plot
		plt.imshow(Pim[::-1],aspect='auto',extent=[Aim[0],Aim[-1],Xim[0],Xim[-1]],interpolation="none")
		
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
		plt.title("Pressure normalised by WN result")
		plot_acco(plt.gca(), xlabel="$\\alpha=(f_0^2\\tau/T\\zeta)^{1/2}$", ylabel="Wall separation")
		plt.grid(None)
		plt.colorbar()
	
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
def filename_pars(filename):
	"""
	Scrape filename for parameters
	"""
	#
	start = filename.find("_a") + 2
	a = float(filename[start:filename.find("_",start)])
	#
	start = filename.find("_X") + 2
	X = float(filename[start:filename.find("_",start)])
	#
	try:
		start = filename.find("_D") + 2
		D = float(filename[start:filename.find("_",start)])
	except ValueError:
		D = 0.0
	#
	try:
		start = filename.find("_dt") + 3
		dt = float(filename[start:filename.find(".npy",start)])
	except ValueError:
		start = filename.find("_dt",start) + 3
		dt = float(filename[start:filename.find(".npy",start)])
	#
	try:
		start = filename.find("_ym") + 2
		ymax = float(filename[start:filename.find("_",start)])
	except ValueError:
		ymax = 0.5
	#
	return a, X, D, dt, ymax

##=============================================================================
def plot_acco(ax, **kwargs):
	"""
	Plot accoutrements.
	kwargs: title, subtitle, xlabel, ylabel, plotfile
	"""
	me = "HopfieldPlotter.plot_acco: "
	try: ax.set_title(kwargs["title"])
	except: pass
	try: ax.suptitle(kawrgs["subtitle"])
	except: pass
	try: ax.set_xlabel(kwargs["xlabel"], fontsize=14)
	except: ax.set_xlabel("$x$ position", fontsize=14)
	try: ax.set_ylabel(kwargs["ylabel"], fontsize=14)
	except: pass
	ax.grid(True)
	try:
		if kwargs["legloc"]!="":	ax.legend(loc=kwargs["legloc"], fontsize=11)
		else: pass
	except KeyError: ax.legend(loc="best", fontsize=11)
	return
	
##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()
