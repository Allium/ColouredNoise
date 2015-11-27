import numpy as np
from matplotlib import pyplot as plt
from sys import argv
import os, glob
import optparse
from time import time as sysT
from LE_LightBoundarySim import lookup_xmax, calculate_xbin
from LE_Pressure import plot_acco

##=============================================================================
def main():
	"""
	Fit PDF to exponential in the wall region.
	
	Reads in EITHER a BHIS file OR a directory of them.
	"""
	me = "Tail.main: "
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
		tail_plot(argv[1],verbose)
	elif os.path.isdir(argv[1]):
		plot_exp_alpha(argv[1],verbose)
	else:
		print me+"you gave me rubbish. Abort."
		exit()
	
	if verbose: print me+"execution time",round(sysT()-t0,2),"seconds"
	if showfig: plt.show()
	return
	
	
##=============================================================================
def tail_plot(histfile, verbose):	
	"""
	Plot tail PDF on log scale and make exponential fit
	"""
	me = "Tail.tail_plot: "
	
	plotfile = os.path.splitext(histfile)[0]+"_t.png"

	## Get alpha and X from filename
	start = histfile.find("_a") + 2
	alpha = float(histfile[start:histfile.find("_",start)])
	start = histfile.find("_X") + 2
	X = float(histfile[start:histfile.find("_",start)])

	H = np.load(histfile)

	ybins = np.linspace(-0.5,0.5,H.shape[0])
	x = calculate_xbin(0.9*X,X,lookup_xmax(X,alpha),H.shape[1]-1)[H.shape[1]/2-1:]
	H = H[:,H.shape[1]/2-1:]

	Hx = np.trapz(H,x=ybins,axis=0)
	Hx = Hx[Hx!=0]; x = x[Hx!=0]

	## Fit; throw away first portion of data
	if alpha<=0.2: lm = Hx.shape[0]/6
	else: lm=Hx.shape[0]/10
	fit = np.polyfit(x[lm:],np.log(Hx[lm:]),1)
	fit_fn = np.poly1d(fit)

	## Plot (x-X distance into wall)
	plt.semilogy(x-X,Hx,label="Data")
	plt.semilogy(x-X,np.exp(fit_fn(x)),"r--",label="$\exp["+str(round(fit[0],1))+"x]$")
	plt.xlim(left=0.0)
	plot_acco(plt.gca(),title="Wall region, $\\alpha="+str(alpha)+"$",
		xlabel="Distance into wall region",ylabel="PDF $p(x)$")
	
	plt.savefig(plotfile)
	if verbose:	print me+"figure saved to",plotfile
	
	return
	
##=============================================================================
def plot_exp_alpha(dirpath, verbose):
	"""
	Plot exponents of fit in wall region against alphs
	"""
	me = "Tail.plot_exp_alpha: "
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
	exponplot = dirpath+"/TailAlpha.png"
	Alpha = np.zeros(numfiles)
	Expon = np.zeros(numfiles)
		
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
		x = calculate_xbin(xmin,X,xmax,H.shape[1]-1)[H.shape[1]/2-1:]
		y = np.linspace(-ymax,ymax,H.shape[0])
		
		H = H[:,H.shape[1]/2-1:]
		## Marginalise to PDF in x and eliminate zeroes
		Hx = np.trapz(H,x=y,axis=0)
		Hx = Hx[Hx!=0]; x = x[Hx!=0]

		## Fit; throw away first portion of data
		if Alpha[i]<=0.2: lm = Hx.shape[0]/6
		else: lm=Hx.shape[0]/10
		fit = np.polyfit(x[lm:],np.log(Hx[lm:]),1)
		Expon[i] = fit[0]
		
	plt.plot(Alpha,Expon,"bo")
	plot_acco(plt.gca(), title="Wall region: exponential tail of PDF",
		xlabel="$\\alpha$", ylabel="Exponent", legloc="")
	
	plt.savefig(exponplot)
	if verbose: print me+"plot saved to",exponplot
	
	return

##=============================================================================
if __name__=="__main__":
	main()