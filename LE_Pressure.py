
import numpy as np
from matplotlib import pyplot as plt
from sys import argv
import os
import optparse
from time import time as sysT

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
	
	BUGS
	
	HISTORY
		2015/11/12	Started
	"""
	me = "LE_Pressure: "
	t0 = sysT()
	
	## Options
	parser = optparse.OptionParser()	
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt = parser.parse_args()[0]
	showfig = opt.showfig
	verbose = opt.verbose
	
	## Need to WALK
	# datafiles = np.sort(glob.glob(argv[1]+"/*.npy"))
	
	filepath = argv[1].replace("\\","/")
	plotfile = "Pressure/"+os.path.splitext(os.path.split(filepath)[1])[0]+"_P.png"

	Hx = np.load(filepath).sum(axis=1)
	xmin,xmax = 0.9,1.1
	x  = np.linspace(xmin,xmax,Hx.shape[0])
	Hx /= np.trapz(Hx,x=x)
	
	## Get alpha from filename
	try:
		start = filepath.find("b=")
		alpha = float(filepath[start+2:filepath.find("/",start)])
	except: alpha = 0.1
	if verbose: print me+"alpha =",alpha
	
	force = -alpha*0.5*(np.sign(x-1)+1)
	press = - np.cumsum(force*Hx)
	
	fig,ax = plt.subplots(1,2)
	ax[0].plot(x,Hx)
	plot_acco(ax[0],ylabel="PDF")
	ax[1].plot(x,press)
	plot_acco(ax[1],ylabel="Pressure")
	plt.tight_layout()
	fig.suptitle("$\\alpha=$"+str(alpha),fontsize=16);plt.subplots_adjust(top=0.9)
	
	plt.savefig(plotfile)
	if verbose: print me+"plot saved to",plotfile
	
	print me+"execution time",round(sysT()-t0,2),"seconds"
	if showfig: plt.show()
	
	
	return
	
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
	try: ax.legend(loc=kwargs["legloc"], fontsize=11)
	except KeyError: ax.legend(loc="best", fontsize=11)
	return
	
##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()