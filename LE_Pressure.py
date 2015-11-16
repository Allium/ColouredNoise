
import numpy as np
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
		python LE_Pressure.py histfile
	
	ARGUMENTS
		histfile	path to density histogram
	
	OPTIONS
	
	FLAGS
		-v --verbose
		-s --show
	
	EXAMPLE
		python LE_Pressure.py dat_LE_stream\b=0.01\BHIS_y0.5bi50r5000b0.01X1seed65438.npy
	
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
	plotfilePDF = "Pressure/"+os.path.splitext(os.path.split(filepath)[1])[0]+"_PDF.png"
	plotfile = "Pressure/"+os.path.splitext(os.path.split(filepath)[1])[0]+"_P.png"
	
	## Get alpha from filename
	start = filepath.find("_a") + 2
	alpha = float(filepath[start:filepath.find("_",start)])
	if verbose: print me+"alpha =",alpha
	
	## Load data and find ranges
	H = np.load(filepath)
	Hx = H.sum(axis=0)
	xmin,xmax = 0.8,calculate_xmax(1.0,alpha)
	x  = np.linspace(xmin,xmax,Hx.shape[0])
	# Hx /= np.trapz(Hx,x=x)
	
	## 2D PDF plot
	if 1:
		plt.imshow(H, extent=[xmin,xmax,-0.5,0.5], aspect="auto")
		plot_acco(plt.gca(),xlabel="$x$",ylabel="$y$",title="$\\alpha=$"+str(alpha))
		plt.savefig(plotfilePDF)
		if verbose: print me+"plot saved to",plotfilePDF

	## Calculate pressure
	force = -alpha*0.5*(np.sign(x-1)+1)
	press = - np.cumsum(force*Hx)
	
	fig,ax = plt.subplots(1,2)
	ax[0].plot(x,Hx)
	ax[0].set_xlim(left=0.9)
	plot_acco(ax[0],ylabel="PDF")
	ax[1].plot(x,press)
	ax[1].set_xlim(left=0.9)
	plot_acco(ax[1],ylabel="Pressure")
	plt.tight_layout()
	fig.suptitle("$\\alpha=$"+str(alpha),fontsize=16);plt.subplots_adjust(top=0.9)
	
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
		
	return plotfilePDF, plotfile
	
##=============================================================================
def pressure_plot_dir(dirpath, verbose):
	"""
	Plot pressure at "infinity" against alpha for all files in directory
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
		
		## Load data and find ranges
		Hx = np.load(filepath).sum(axis=0)
		xmin,xmax = 0.8,calculate_xmax(1.0,Alpha[i])
		x  = np.linspace(xmin,xmax,Hx.shape[0])
		Hx /= np.trapz(Hx,x=x)

		## Calculate pressure
		force = -Alpha[i]*0.5*(np.sign(x-1)+1)
		Press[i] = -np.sum(force*Hx)
	
	plt.plot(Alpha,Press,"bo")
	plot_acco(plt.gca(), xlabel="$\\alpha$", ylabel="Pressure")
	
	plt.savefig(pressplot)
	if verbose: print me+"plot saved to",pressplot
	
	return pressplot


##=============================================================================
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