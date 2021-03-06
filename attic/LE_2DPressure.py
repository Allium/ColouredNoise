
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
from scipy import integrate
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sys import argv
import os, glob
import optparse
import warnings
from time import time as sysT

from LE_Utils import save_data, filename_pars
from LE_LightBoundarySim import lookup_xmax, calculate_xmin, calculate_xini,\
				calculate_xbin, calculate_ybin
from LE_2DLBS import force_2D

warnings.filterwarnings("ignore",
	"No labelled objects found. Use label='...' kwarg on individual plots.",
	UserWarning)
warnings.filterwarnings("ignore",
	"invalid value encountered in sign",
	RuntimeWarning)

## Global variables
from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

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
		Using masked array breaks contour plot.
	
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
	parser.add_option('--rawp',
		dest="rawp", default=False, action="store_true")
	parser.add_option('-a','--plotall',
		dest="plotall", default=False, action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")		
	opt, arg = parser.parse_args()
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	rawp	= opt.rawp
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
	plotfile = os.path.dirname(histfile)+"/PDFP"+os.path.basename(histfile)[4:-4]+".png"
	
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
	c = circle_centre(X,R,ymax)
	
	## Space (for axes)
	try:
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		xbins = bins["xbins"]
		ybins = bins["ybins"]
		xini = xbins[0]
		xmax = xbins[-1]
	except (IOError, KeyError):
		xini = calculate_xini(X,alpha)
		xmax = lookup_xmax(c[0]+R,alpha)
		xbins = calculate_xbin(xini,X,xmax,H.shape[1])
		ybins = calculate_ybin(0.0,ymax,H.shape[0]+1)
	x = 0.5*(xbins[1:]+xbins[:-1])
	y = 0.5*(ybins[1:]+ybins[:-1])
	
	## Set up plot
	fig,axs = plt.subplots(1,2)
		
	## pdf plot
	ax = axs[0]
	H[:,0]=H[:,1]
	Xm,Ym = np.meshgrid(x,y)
	CS = ax.contourf(Xm,Ym[::-1],H,10)
	
	## Colourbar
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("top", size="5%", pad=0.4)
	cbar = fig.colorbar(CS, cax=cax, ax=ax, orientation="horizontal",
		use_gridspec=True, ticks=[H.min(),H.mean(),H.max()])
	cbar.ax.set_xticklabels(["Low", "Mean", "High"])
	### http://stackoverflow.com/questions/13310594/positioning-the-colorbar
	## Plot curved wall
	wallx = np.linspace(X,c[0]+R,201)
	wally = c[1]+np.sqrt(R*R-(wallx-c[0])**2)
	ax.plot(wallx,wally, "r--",linewidth=2)
	## Accoutrements
	ax.set_xlim([xini,xmax])
	ax.set_ylim([0.0,ymax])
	ax.set_xlabel("$x$", fontsize=fsa)
	ax.set_ylabel("$y$", fontsize=fsa)
		
	## Calculate force array (2d)
	force = -1.0 * ( (Xm-c[0])**2 + (Ym-c[1])**2 > R*R ) * ( Xm-c[0]>0.0 )
	## Pressure array (2d) -- sum rather than trapz
	press = -1.0*(force*H).sum(axis=0).cumsum(axis=0)
	
	## Pressure plot
	ax = axs[1]
	ax.plot(x,press,label="CN simulation")
	## Bulk and wall regions
	ax.axvspan(xini,X, color="b",alpha=0.1) 
	ax.axvspan(X,c[0]+R, color="m",alpha=0.05)
	ax.axvspan(R,xmax, color="r",alpha=0.05)
	## Ideal gas result
	ax.hlines(pressure_IG(X,R,ymax,alpha),xini,xmax,linestyle="-",color="g",label="WN theory") 
	ax.hlines(0.5/ymax/(1.0+X-xini),xini,xmax,linestyle="--",color="g",label="WN flat theory")
	## Accoutrements
	ax.set_xlim([xini,xmax])
	ax.set_xlabel("$x$", fontsize=fsa)
	ax.set_ylabel("Pressure", fontsize=fsa)
	ax.grid()
	ax.legend(loc="best",fontsize=fsl)
	
	## Tidy figure
	fig.suptitle(os.path.basename(plotfile),fontsize=fst)
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
	
	## File discovery
	histfiles = np.sort(glob.glob(dirpath+"/BHIS_2D_*.npy"))
	numfiles = len(histfiles)
	if verbose: print me+"found",numfiles,"files"
	
	## Initialise
	Alpha = np.zeros(numfiles) 
	X = np.zeros(numfiles)
	R = np.zeros(numfiles)
	Press = np.zeros(numfiles)
	xini = np.zeros(numfiles)
		
	## Loop over files
	for i,histfile in enumerate(histfiles):
	
		## Get pars from filename
		pars = filename_pars(histfile)
		[Alpha[i],X[i],D,dt,ymax,R[i]] = [pars[key] for key in ["a","X","D","dt","ymax","R"]]
		assert (R[i] is not None), me+"You are using the wrong program. R should be defined."
		assert (D == 0.0), me+"Cannot yet handle soft potential. D should be 0.0."

		## Load data and normalise
		H = np.load(histfile)
		H /= H.sum()
		
		## Centre of circle for curved boundary
		c = circle_centre(X[i],R[i],ymax)
		
		## Space (for axes)
		try:
			bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
			xbins = bins["xbins"]
			ybins = bins["ybins"]
			xini[i] = xbins[0]
			xmax = xbins[-1]
		except (IOError, KeyError):
			xini[i] = calculate_xini(X[i],Alpha[i])
			xmax = lookup_xmax(c[0]+R[i],Alpha[i])
			xbins = calculate_xbin(xini,X[i],xmax,H.shape[1])
			ybins = calculate_ybin(0.0,ymax,H.shape[0]+1)
		x = 0.5*(xbins[1:]+xbins[:-1])	
		y = 0.5*(ybins[1:]+ybins[:-1])
	
		## Calculate force array (2D)
		Xm,Ym = np.meshgrid(x,y)
		force = -1.0 * ( (Xm-c[0])**2 + (Ym-c[1])**2 > R[i]*R[i] ) * ( Xm-c[0]>0.0 )
		## Pressure array (2d) -- sum rather than trapz
		Press[i] = -1.0*(force*H).sum(axis=0).cumsum(axis=0)[-1]
	
	## ------------------------------------------------	
	## Create 3D pressure array and 1D a,X,R coordinate arrays

	## Ordered independent variable arrays
	AA = np.unique(Alpha)
	XX = np.unique(X)
	RR = np.unique(R)
	
	## 3D pressure array: [X,R,A]
	PP = np.zeros([XX.size,RR.size,AA.size])
	PPWN = np.zeros(PP.shape)
	for i in range(XX.size):
		Xidx = (X==XX[i])
		for j in range(RR.size):
			Ridx = (R==RR[j])
			for k in range(AA.size):
				Aidx = (Alpha==AA[k])
				Pidx = Xidx*Ridx*Aidx
				try: PP[i,j,k] = Press[Pidx]
				except ValueError: pass
				PPWN[i,j,k] = pressure_IG(XX[i],RR[j],xini[Pidx],ymax,AA[k])
	
	## Normalise by WN result
	if 1: PP /= PPWN
	
	## Mask zeros
	PP = np.ma.array(PP, mask = PP==0.0)
	
	## ------------------------------------------------
	## 1D plots
	
	## Which plots to make (abcissa,multiline,subplot,dimension)
	[ARX1,AXR1,XAR1,XRA1,RXA1,RAX1] = [1,0,0,0,0,0]
	
	if ARX1:
		fig, axs = plt.subplots(1,2,sharey=True)
		for i in range(RR.size):
			axs[0].plot(AA,PP[0,i,:],  "o-", label="$R = "+str(RR[i])+"$") 
			axs[1].plot(AA,PP[-1,i,:], "o-", label="$R = "+str(RR[i])+"$")
		for j in range(len(axs)):
			axs[j].set_xlim((AA[0],AA[-1]))
			axs[j].set_ylim((0.0,np.array([PP[0,:,:],PP[-1,:,:]]).max()))
			axs[j].set_xlabel("$\\alpha$",fontsize=fsa)
			axs[j].set_title("$X = "+str(XX[0-j])+"$",fontsize=fsa)
			axs[j].grid()
		axs[0].set_ylabel("Pressure",fontsize=fsa)
		axs[1].legend(loc="best",fontsize=fsl)
		plt.tight_layout()
		pressplot = dirpath+"/PARX1_dt"+str(dt)+".png"
		plt.savefig(pressplot)
		if verbose: print me+"plot saved to",pressplot
	
	if AXR1:
		fig, axs = plt.subplots(1,2,sharey=True)
		for i in range(XX.size):
			axs[0].plot(AA,PP[i, 0,:], "o-", label="$x_{\\rm wal} = "+str(XX[i])+"$") 
			axs[1].plot(AA,PP[i,-1,:], "o-", label="$x_{\\rm wal} = "+str(XX[i])+"$")
		for j in range(len(axs)):
			axs[j].set_xlim((AA[0],AA[-1]))
			axs[j].set_ylim((0.0,np.array([PP[:,0,:],PP[:,-1,:]]).max()))
			axs[j].set_xlabel("$\\alpha$",fontsize=fsa)
			axs[j].set_title("$R = "+str(RR[0-j])+"$",fontsize=fsa)
			axs[j].grid()
		axs[0].set_ylabel("Pressure",fontsize=fsa)
		axs[1].legend(loc="best",fontsize=fsl)
		plt.tight_layout()
		pressplot = dirpath+"/PAXR1_dt"+str(dt)+".png"
		plt.savefig(pressplot)
		if verbose: print me+"plot saved to",pressplot	

	## ------------------------------------------------
	## 2D plots
	
	## Which plots to make (abcissa,ordinate,subplot,dimension)
	[ARX2,AXR2,XAR2,XRA2,RXA2,RAX2] = [0,0,0,0,0,0]
	
	if ARX2:
		fig, axs = plt.subplots(1,2,sharey=True)
		for i in range(RR.size):
			axs[0].contourf(AA,RR,PP[0,:,:],  vmin=0.0) 
			axs[1].contourf(AA,RR,PP[-1,:,:], vmin=0.0)
		for j in range(len(axs)):
			axs[j].set_xlim((AA[0],AA[-1]))
			axs[j].set_ylim((RR[0],RR[-1]))
			axs[j].set_xlabel("$\\alpha$",fontsize=fsa)
			axs[j].set_title("$X = "+str(X[0-j])+"$",fontsize=fsa)
		axs[0].set_ylabel("$R$",fontsize=fsa)
		plt.tight_layout()
		pressplot = dirpath+"/PARX2_dt"+str(dt)+".png"
		plt.savefig(pressplot)
		if verbose: print me+"plot saved to",pressplot
	
	
	## ------------------------------------------------	
		
	return


##=============================================================================
def circle_centre(X,R,ymax):
	return [X-np.sqrt(R*R-ymax*ymax),0.0]

	
##=============================================================================
def pressure_IG(X,R,xini,ym,a):
	"""
	Theoretical pressure of a white noise gas.
	See notes 22/02/2016
	"""
	cx = circle_centre(X,R,ym)[0] - xini
	return 1.0/(2*ym*(1+cx)+ym*np.sqrt(R*R-ym*ym)+R*R*np.arcsin(ym/R))


##=============================================================================
if __name__=="__main__":
	main()
