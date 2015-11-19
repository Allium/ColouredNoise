
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import integrate
from matplotlib import pyplot as plt
from sys import argv
import os, glob
import optparse
from time import time as sysT
from LE_LightBoundarySim import calculate_xmax

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
	"""
	me = "LE_Pressure.main: "
	t0 = sysT()
	
	## Options
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-X','--wallpos',
                  dest="X",default=1.0,type="float")	
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")		
	opt = parser.parse_args()[0]
	if opt.help: print main.__doc__; return
	X		= opt.X
	showfig = opt.showfig
	verbose = opt.verbose
	
	argv[1] = argv[1].replace("\\","/")
	if os.path.isfile(argv[1]):
		pressure_pdf_plot_file(argv[1],X,verbose)
	elif os.path.isdir(argv[1]):
		pressure_plot_dir(argv[1],X,verbose)
	else:
		print me+"you gave me rubbish. Abort."
		exit()
	
	if verbose: print me+"execution time",round(sysT()-t0,2),"seconds"
	if showfig: plt.show()
	
	return
	
##=============================================================================
def pressure_pdf_plot_file(filepath, X, verbose):
	"""
	Make plot for a single file
	"""
	me = "LE_Pressure.pressure_pdf_plot_file: "
	t0 = sysT()
	
	## Filenames
	plotfile = os.path.splitext(filepath)[0]+"_P.png"
	plotfilePDF = os.path.splitext(filepath)[0]+"_PDF.png"
	
	## Get alpha and X from filename
	start = filepath.find("_a") + 2
	alpha = float(filepath[start:filepath.find("_",start)])
	start = filepath.find("_X") + 2
	X = float(filepath[start:filepath.find("_",start)])
	if verbose: print me+"alpha =",alpha
	
	## Load data
	H = np.load(filepath)
	
	## Space
	xmin,xmax = 0.8*X,calculate_xmax(X,alpha)
	ymax = 0.5
	x = np.linspace(xmin,xmax,H.shape[1])
	y = np.linspace(-ymax,ymax,H.shape[0])
	
	## Marginalise to PDF in X
	Hx = np.trapz(H,x=y,axis=0)
	
	## 2D PDF plot
	if 1:
		plt.imshow(H, extent=[xmin,xmax,-ymax,ymax], aspect="auto")
		plot_acco(plt.gca(),xlabel="$x$",ylabel="$\\eta$",title="$\\alpha=$"+str(alpha))
		plt.savefig(plotfilePDF)
		if verbose: print me+"plot saved to",plotfilePDF

	## Calculate pressure
	force = -alpha*0.5*(np.sign(x-X)+1)
	press = pressure(force,Hx,x,"discrete")
	
	fig,ax = plt.subplots(1,2)
	ax[0].plot(x,Hx)
	ax[0].set_xlim(left=0.8*X)
	plot_acco(ax[0],ylabel="PDF p(x)")
	ax[1].plot(x,press)
	ax[1].set_xlim(left=0.8*X)
	plot_acco(ax[1],ylabel="Pressure")
	plt.tight_layout()
	fig.suptitle("$\\alpha=$"+str(alpha),fontsize=16);plt.subplots_adjust(top=0.9)
	
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
		
	return fig
	
##=============================================================================
def pressure_plot_dir(dirpath, X, verbose):
	"""
	Plot pressure at "infinity" against alpha for all files in directory.
	
	Be careful heed changes in parameters between files in directory
	"""
	me = "LE_Pressure.pressure_plot_dir: "
	t0 = sysT()
	
	## FIle discovery
	histfiles = np.sort(glob.glob(dirpath+"/*.npy"))
	numfiles = len(histfiles)
	if verbose: print me+"found",numfiles,"files"
	
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
		xmin,xmax = 0.8*X,calculate_xmax(X,Alpha[i])
		ymax = 0.5
		x = np.linspace(xmin,xmax,H.shape[1])
		y = np.linspace(-ymax,ymax,H.shape[0])
		
		## Marginalise to PDF in X
		Hx = np.trapz(H,x=y,axis=0)

		## Calculate pressure
		force = -Alpha[i]*0.5*(np.sign(x-X)+1)
		Press[i] = -np.sum(force*Hx)
	
	plt.plot(Alpha,Press,"bo")
	plot_acco(plt.gca(), xlabel="$\\alpha$", ylabel="Pressure")
	
	plt.savefig(pressplot)
	if verbose: print me+"plot saved to",pressplot
	
	return pressplot


##=============================================================================
##=============================================================================
def pressure(force,Hx,x, method="interp_prod"):
	"""
	Calculate the pressure given an array of forces and densities at positions x.
	Returns an array of pressure value at every point in x.
	"""
	me = "LE_Pressure.pressure: "
	
	if method is "discrete":
		# press = -(x[1]-x[0])*np.cumsum(force*Hx)
		press = np.array([np.trapz((-force*Hx)[:i], x[:i]) for i,xi in enumerate(x)])
	elif method is "interp_prod":
		## Very slow
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
	# try: ax.legend(loc=kwargs["legloc"], fontsize=11)
	# except KeyError: ax.legend(loc="best", fontsize=11)
	return
	
##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()