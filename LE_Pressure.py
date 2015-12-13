
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
		-v --verbose
		-s --show
	
	EXAMPLE
		python LE_Pressure.py dat_LE_stream\b=0.01\BHIS_y0.5bi50r5000b0.01X1seed65438.npy
		
	NOTES
	
	BUGS / TODO
		-- 2D PDF doen't work with unequal bin width
	
	HISTORY
		12 November 2015	Started
		15 November 2015	Pressure versus alpha functionality
		30 November 2015	Added IG result
	"""
	me = "LE_Pressure.main: "
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
	
	argv[1] = argv[1].replace("\\","/")
	if os.path.isfile(argv[1]):
		pressure_pdf_plot_file(argv[1],verbose)
	elif os.path.isdir(argv[1]):
		pressure_plot_dir(argv[1],verbose)
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
	
	D = 0.01
	
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
	Hx /= np.trapz(Hx,x=x,axis=0)
	
	## Calculate pressure
	force = force_x(x,alpha,X,D)
	press = pressure_x(force,Hx,x)
	xIG, forceIG, HxIG, pressIG = ideal_gas(alpha, x, X, D, xmin, dt)
		
	fig,ax = plt.subplots(1,2)
	ax[0].plot(x,Hx,label="Simulation")
	ax[0].plot(xIG,HxIG,"r--",label="White noise")
	ax[0].plot(xIG,-forceIG,"m:",linewidth=2,label="Force")
	ax[0].set_xlim(left=xmin,right=xmax)
	ax[0].set_ylim(bottom=0.0,top=np.ceil(Hx[Hx.shape[0]/4:].max()))
	plot_acco(ax[0],ylabel="PDF p(x)",legloc="best")
	ax[1].plot(x,press,label="")
	ax[1].plot(xIG,pressIG,"r--",label="")
	# ax[1].plot([xmin,xmax],[dt/(X-xmin+dt/alpha)]*2, "g--")
	ax[1].set_xlim(left=xmin,right=xmax)
	plot_acco(ax[1], ylabel="Pressure", legloc="")
	plt.tight_layout()
	fig.suptitle("$\\alpha=$"+str(alpha)+", $\\Delta=$"+str(D),fontsize=16)
	plt.subplots_adjust(top=0.9)
	
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
		
	return fig
	
##=============================================================================
def pressure_plot_dir(dirpath, verbose):
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
		force = force_x(x,Alpha[i],X,0.01)
		Press[i] = np.trapz(-force*Hx, x)/dt
		#PressIG[i] = dt/(X-xmin+dt/Alpha[i])
	
	## Sort values
	sortind = np.argsort(Alpha)
	Alpha = Alpha[sortind]; Press = Press[sortind]; PressIG = PressIG[sortind]
	
	## Calculate IG pressure on a finer grid -- assume X, dt same
	AlphaIG = np.arange(0.01,Alpha[-1],0.01)
	PressIG = 1/(X-xmin+dt/AlphaIG)
		
	## Linear fit for CN pressure
	linfit = [round(elem,2) for elem in np.polyfit(Alpha[5:-4], Press[5:-4], 1)]
	linfit_fn = np.poly1d(linfit)
	
	## IG-type fit
	# from scipy.optimize import curve_fit
	# fitfunc = lambda z,m,A: A*z/(z*(X-xmin)+m)
	# popt, pcov = curve_fit(fitfunc, Alpha, Press)
	
	## Plotting
	plt.errorbar(Alpha,Press,yerr=0.05,fmt='bo', ecolor='grey', capthick=2,label="Simulated")
	# plt.plot(Alpha,Press,"bo",label="Simulated")
	plt.plot(Alpha[5:-4],linfit_fn(Alpha[5:-4]),"b--",label="$P="+str(linfit[0])+"\\alpha+"+str(linfit[1])+"$")
	plt.plot(AlphaIG,PressIG,"r-",label="White noise")
	plt.ylim(bottom=0.0)
	plot_acco(plt.gca(), xlabel="$\\alpha=f_0^2\\tau/T\\zeta$", ylabel="Pressure")
	
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
def ideal_gas(alpha, x, X, D, xmin, dt):
	"""
	Calculate PDF and pressure for ideal gas
	Upsample space
	"""
	## Upsampling
	up = 6
	xbinsIG = calculate_xbin(x[0],X,x[-1],up*len(x))
	xIG = 0.5*(xbinsIG[1:]+xbinsIG[:-1])
	forceIG = force_x(xIG,alpha,X,D)
	## Predicted solution
	UIG = np.array([np.trapz((-forceIG)[:i], x=xIG[:i]) for i in range(len(xIG))])
	HxIG = np.exp(-UIG/dt)
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
	start = filename.find("_D") + 2
	D = float(filename[start:filename.find("_",start)])
	#
	start = filename.find("_dt") + 3
	dt = float(filename[start:filename.find(".npy",start)])
	#
	try:
		start = filename.find("_ym") + 2
		ymax = float(filename[start:filename.find("_",start)])
	except:
		ymax = 0.5
	#
	return a, X, D, dt, ymax
	
##=============================================================================
def av_pd(p,x,x1,x2,fac=0.02):
	"""
	Average the probability density between two points
	"""
	ind1 = np.abs(x-x1).argmin()
	ind2 = np.abs(x-x2).argmin()
	return p[ind1:ind2].mean()

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