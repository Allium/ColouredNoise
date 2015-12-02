
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
		-- Honest normalisation -- affects pressure
	
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
	
	## Filenames
	plotfile = os.path.splitext(filepath)[0]+"_P.png"
	plotfilePDF = os.path.splitext(filepath)[0]+"_PDF.png"
	
	## Get alpha and X from filename
	alpha, X, dt = filename_pars(filepath)
	if verbose: print me+"alpha =",alpha,"and X =",X
	
	## Load data
	H = np.load(filepath)
	
	## Space
	xmin,xmax = 0.9*X,lookup_xmax(X,alpha)
	ymax = 0.5
	ybins = np.linspace(-ymax,ymax,H.shape[0]+1)
	y = 0.5*(ybins[1:]+ybins[:-1])
	xbins = calculate_xbin(xmin,X,xmax,H.shape[1])
	x = 0.5*(xbins[1:]+xbins[:-1])
	
	## Marginalise to PDF in x
	Hx = np.trapz(H,x=y,axis=0)
	# Hx = H.sum(axis=0) * (y[1]-y[0])	## Should be dot product with diffy
	if verbose: print me+"integral of density",np.trapz(Hx,x=x,axis=0)
	
	## 2D PDF plot
	if 0:
		plt.imshow(H, extent=[xmin,xmax,-ymax,ymax], aspect="auto")
		plot_acco(plt.gca(),xlabel="$x$",ylabel="$\\eta$",title="$\\alpha=$"+str(alpha))
		plt.savefig(plotfilePDF)
		if verbose: print me+"2D PDF plot saved to",plotfilePDF

	## Calculate pressure
	force = -alpha*0.5*(np.sign(x-X)+1)
	press = pressure_x(force,Hx,x,"discrete")
	Hx_wn = 0.5*np.exp(-50*alpha*(x-X)); Hx_wn[:len(x)/2]=Hx_wn[len(x)/2]	## Note quite right
	pressIG = pressure_x(force,Hx_wn,x,"discrete")
	
	fig,ax = plt.subplots(1,2)
	ax[0].plot(x,Hx,label="")
	ax[0].plot(x,Hx_wn,"r--",label="")
	ax[0].set_xlim(left=xmin,right=xmax)
	ax[0].set_ylim(bottom=0.0,top=np.ceil(Hx[Hx.shape[0]/4:].max()))
	plot_acco(ax[0],ylabel="PDF p(x)")
	ax[1].plot(x,press,label="")
	ax[1].plot(x,pressIG,"r--",label="")
	ax[1].set_xlim(left=xmin,right=xmax)
	plot_acco(ax[1], ylabel="Pressure", legloc="")
	plt.tight_layout()
	fig.suptitle("$\\alpha=$"+str(alpha),fontsize=16);plt.subplots_adjust(top=0.9)
	
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
	
	## Assume all files have same X
	start = histfiles[0].find("_X") + 2
	X = float(histfiles[0][start:histfiles[0].find("_",start)])
	if verbose: print me+"determined X="+str(X)
	
	## Outfile name
	pressplot = dirpath+"/PressureAlpha.png"
	Alpha = np.zeros(numfiles)
	Press = np.zeros(numfiles)
		
	## Loop over files
	for i,filepath in enumerate(histfiles):
		
		## Find alpha
		start = filepath.find("_a") + 2
		Alpha[i] = float(filepath[start:filepath.find("_",start)])
				
		## Load data
		H = np.load(filepath)
		
		## Space
		xmin,xmax = 0.9*X,lookup_xmax(X,Alpha[i])
		ymax = 0.5
		x = calculate_xbin(xmin,X,xmax,H.shape[1]-1)
		y = np.linspace(-ymax,ymax,H.shape[0])
		
		## Marginalise to PDF in X
		Hx = np.trapz(H,x=y,axis=0)

		## Calculate pressure
		force = -Alpha[i]*0.5*(np.sign(x-X)+1)
		Press[i] = np.trapz(-force*Hx, x)
		
	plt.plot(Alpha,Press,"bo",label=".")
	plt.ylim(bottom=0.0)
	plot_acco(plt.gca(), xlabel="$\\alpha$", ylabel="Pressure", legloc="")
	
	plt.savefig(pressplot)
	if verbose: print me+"plot saved to",pressplot
	
	return pressplot


##=============================================================================
##=============================================================================
def pressure_x(force,Hx,x, method="interp_prod"):
	"""
	Calculate the pressure given an array of forces and densities at positions x.
	Returns an array of pressure value at every point in x.
	"""
	me = "LE_Pressure.pressure: "
	
	if method is "discrete":
		press = np.array([np.trapz((-force*Hx)[:i], x[:i]) for i,xi in enumerate(x)])
	elif method is "interp_prod":
		## Very slow; issues with discontinuity
		# dpress = UnivariateSpline(x,-force*Hx,k=1)
		# press = np.array([dpress.integral(x[0],xi) for xi in x])
		dpress = interp1d(x,-force*Hx)
		# press = np.array([integrate.quad(dpress,x[0],xi, points=[float(xi>1.0)] )[0]\
			# for xi in x])
		press =  [integrate.quad(dpress,x[0],xi )[0]	for xi in x[x<1.0]]
		press += [integrate.quad(dpress,x[0],xi )[0]	for xi in x[x>1.0]]
		press = np.array(press)
	elif method is "interp_both":
		## Not tested
		force = UnivariateSpline(x,force,k=2)
		Hx = UnivariateSpline(x,Hx)
		press = -np.array([(force*Hx).integral(x[0],xi) for xi in x])
		
	return press
	
##=============================================================================
def filename_pars(filename):
	"""
	Scrape filename for parameters
	"""
	start = filepath.find("_a") + 2
	a = float(filepath[start:filepath.find("_",start)])
	start = filepath.find("_X") + 2
	X = float(filepath[start:filepath.find("_",start)])
	start = filepath.find("_dt") + 3
	dt = float(filepath[start:filepath.find(".npy",start)])
	return a, X, dt
	
##=============================================================================
def av_pd(p,x,x0,X,fac=0.02):
	"""
	Average the probability density between two points
	"""
	x1 = x0 + fac*X
	x2 = X  - fac*X
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