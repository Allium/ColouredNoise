
import numpy as np
from matplotlib import pyplot as plt
import os, glob, optparse
import warnings
from time import time as sysT
from itertools import chain

from LE_LightBoundarySim import lookup_xmax,calculate_xmin,calculate_xini
from LE_Utils import force_1D_const, force_1D_lin
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
	parser.add_option('--rawp',
		dest="rawp", default=False, action="store_true")
	parser.add_option('-h','--help',
        dest="help",default=False,action="store_true")		
	opt, args = parser.parse_args()
	if opt.help: print main.__doc__; return
	showfig = opt.showfig
	verbose = opt.verbose
	plotall = opt.plotall
	twod 	= opt.twod
	normIG	= not opt.rawp
	
	args[0] = args[0].replace("\\","/")
	if plotall and os.path.isdir(args[0]):
		showfig = False
		allfiles(args[0],verbose)
	if os.path.isfile(args[0]):
		pressure_pdf_plot_file(args[0],verbose)
	elif os.path.isdir(args[0]):
		pressure_plot_dir(args[0],verbose, twod, normIG)
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
	[alpha,X,D,R,ftype] = [pars[key] for key in ["a","X","D","R","ftype"]]
	assert (R is None), me+"You are using the wrong program. R should not enter."
	force_x = force_1D_const if ftype is "const" else force_1D_lin
	if verbose: print me+"[a, X, D, ftype] =",[alpha,X,D,ftype]
	
	## Load data
	H = np.load(histfile)
	# H[:,0]=H[:,1]
	
	## Space, for axes
	bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
	xbins = bins["xbins"]
	ebins = bins["ebins"]
	xmin = xbins[0]
	xmax = xbins[-1]
	xini = 0.5*(xmin+X)
	x = 0.5*(xbins[1:]+xbins[:-1])	
	e = 0.5*(ebins[1:]+ebins[:-1])
		
	## Marginalise to PDF in x
	Hx = np.trapz(H,x=e,axis=0)
	Hx /= np.trapz(Hx,x=x)	
	
	## Calculate pressure
	force = force_x(x,X,D)
	press = pressure_x(force,Hx,x)
	xIG, forceIG, HxIG, pressIG = ideal_gas(x, X, D, force_x)
	
	## PLOTTING
	fig,axs = plt.subplots(2,1,sharex=True)
	
	## Density plot
	ax = axs[0]
	## Wall
	plot_wall(ax, ftype, x, X)
	##### Will have to fix when D!=0
	# ax.plot(xIG,-forceIG,"m:",linewidth=2,label="Force")
	##
	ax.plot(x,Hx,"b-",label="Simulation")
	ax.plot(xIG,HxIG,"r-",label="White noise")
	ax.set_xlim(left=xini,right=max(xmax,xIG[-1]))
	ax.set_ylim(bottom=0.0,top=1.1)
	ax.set_ylabel("PDF $\\rho(x)$",fontsize=fsa)
	ax.grid()
	ax.legend(loc="upper right",fontsize=fsl)
	
	## Pressure plot
	ax = axs[1]
	## Wall
	plot_wall(ax, ftype, x, X)
	##
	ax.plot(x,press,"b-",linewidth=1, label="CN")
	ax.axhline(press[-1],color="b",linestyle="--",linewidth=1)
	ax.plot(xIG,pressIG,"r-",label="WN")
	ax.axhline(pressIG[-1],color="r",linestyle="--",linewidth=1)
	# ax.axhline(1/(1.0-np.exp(X-xmax)+X-xmin),color="r",linestyle="--",linewidth=1)
	ax.set_xlim(left=xbins[0],right=xbins[-1])
	ax.set_ylim(bottom=0.0, top=np.ceil(max([press[-1],pressIG[-1]])))
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
	
	ftype = filename_pars(histfiles[0])["ftype"]
	force_x = force_1D_const if ftype is "const" else force_1D_lin
	
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
		[Alpha[i],X[i],D,ymax,R] = [pars[key] for key in ["a","X","D","ymax","R"]]
		assert (R is None), me+"You are using the wrong program. R should not enter."
				
		## Load data
		H = np.load(histfile)
		H[:,0]=H[:,1]
		
		## Space
		bins = np.load(os.path.dirname(histfile)+"/BHISBIN"+os.path.basename(histfile)[4:-4]+".npz")
		xbins = bins["xbins"]
		ebins = bins["ebins"]
		xmin[i] = xbins[0]
		xmax[i] = xbins[-1]
		xini = 0.5*(xmin[i]+X[i])
		x = 0.5*(xbins[1:]+xbins[:-1])	
		e = 0.5*(ebins[1:]+ebins[:-1])
		
		## Marginalise to PDF in x and normalise
		Hx = np.trapz(H,x=e,axis=0)
		Hx /= np.trapz(Hx,x=x,axis=0)

		## Calculate pressure
		force = force_x(x,X[i],D)
		Press[i] = np.trapz(-force*Hx, x)
	
	## ----------------------------------------------------
	## Sort values
	sortind = np.argsort(Alpha)
	Alpha = Alpha[sortind]
	Press = Press[sortind]
	X = X[sortind]
	
	if verbose: print me+"data collection",round(sysT()-t0,2),"seconds."
	
	pressplot = dirpath+"/PAX.png"
	
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
			
		## Calculate IG on finer grid, assuming same X, xmin
		AAIG = AA
		PPIG = [[]]*Ncurv
		if D==0.0:
			if ftype is "const":
				PPIG = [1.0/(1.0-np.exp(-4.0)+XX[i]-calculate_xmin(XX[i],AA[i])) for i in range(Ncurv)]
			elif ftype is "linear":
				PPIG = [1.0/(np.sqrt(np.pi/2)-np.exp(-4.0)+XX[i]-calculate_xmin(XX[i],AA[i])) for i in range(Ncurv)]
		else:
			## Needs update!
			raise AttributeError, me+"no can do."
			PPIG = [ideal_gas(a,x,X,D)[3][-1] for a in AAIG]
			
		if normIG: PP = [PP[i]/PPIG[i] for i in range(Ncurv)]
			
		for i in range(Ncurv):
			plt.plot(AA[i], PP[i], 'o-', label=labels[i])
			if not normIG: plt.axhline(PressIG[i], color=plt.gca().lines[-1].get_color(), linestyle="--")
		# if ftype is "linear": plt.plot(AA[0],1/(0.5*AA[0]+1),"m--",label="$(\\alpha/2+1)^{-1}$")
		plt.xlim(right=max(chain.from_iterable(AA)))
		plt.ylim(bottom=0.0)
		plt.title("Pressure normalised by WN result",fontsize=fst)
		xlabel = "$\\alpha=f_0^2\\tau/T\\zeta$" if ftype is "const" else "$\\alpha=k\\tau/\\zeta$"
		plt.xlabel(xlabel,fontsize=fsa)
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
		PIGim = -np.ones((Xim.shape[0],Aim.shape[0]))
		for k in range(Press.shape[0]):
			for i in range(Xim.shape[0]):
				for j in range(Aim.shape[0]):
					if Aim[j]==Alpha[k] and Xim[i]==X[k]:
						Pim[i,j]=Press[k]
						PIGim[i,j]=1.0/(1.0-np.exp(-4.0)+Xim[i]-calculate_xini(Xim[i],Aim[j]))
		## Mask zeros
		Pim = np.ma.array(Pim, mask = Pim<0.0)
		PIGim = np.ma.array(PIGim, mask = PIGim<0.0)
		
		## Normalise by WN reasult
		if normIG:	Pim /= PIGim
						
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
		xlabel = "$\\alpha=f_0^2\\tau/T\\zeta$" if ftype is "const" else "$\\alpha=k\\tau/\\zeta$"
		plt.xlabel(xlabel,fontsize=fsa)
		plt.ylabel("Wall separation",fontsize=fsa)
		plt.grid(False)
		
		# ticks = np.array([0.0,1.0,Pim.min(),Pim.mean(),Pim.max()])
		# tckidx = np.argsort(ticks)
		# ticks = ticks[tckidx]
		# ticklabels = np.array(["0","1","Low", "Mean", "High"])[tckidx]
		# cbar = plt.colorbar(im, ticks=ticks, orientation="vertical")
		# cbar.ax.set_yticklabels(ticklabels, fontsize=fsl)	
		
		# cbar = plt.colorbar(im, ticks=[Pim.min(),Pim.mean(),Pim.max()], orientation="vertical")
		# cbar.ax.set_yticklabels(["Low", "Mean", "High"], fontsize=fsl)
		cbar = plt.colorbar(im, orientation="vertical")
	
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
	press = np.array([np.trapz((-force*Hx)[:i], x=x[:i]) for i,xi in enumerate(x)])
	return press
	
##=============================================================================
def ideal_gas(x, X, D, force_x):
	"""
	Calculate PDF and pressure for ideal gas
	"""
	up=2
	xbinsIG = np.linspace(x[0],x[-1],up*len(x)+1)
	xIG = 0.5*(xbinsIG[1:]+xbinsIG[:-1])
	forceIG = force_x(xIG,X,D)
	## Predicted solution
	UIG = np.array([np.trapz((-forceIG)[:i], x=xIG[:i]) for i in range(len(xIG))])
	HxIG = np.exp(-UIG)
	HxIG /= np.trapz(HxIG,x=xIG)
	## Pressure
	pressIG = pressure_x(forceIG,HxIG,xIG)
	return xIG, forceIG, HxIG, pressIG

	
	
##=============================================================================

def plot_wall(ax, ftype, r, R):
	if ftype is "linear":
		Ridx = np.argmin(np.abs(R-r))
		ax.plot(r,np.hstack([np.zeros(Ridx),r[Ridx:]-R]),"k--",label="Wall")
	else:
		ax.axvline(R,c="k",ls="--",label="Wall")
	return

	
##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()
