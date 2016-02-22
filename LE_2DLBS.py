import numpy as np
import matplotlib.pyplot as plt
import os, time
from datetime import datetime
import optparse, subprocess
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
from sys import argv

from LE_Utils import save_data

from LE_LightBoundarySim import check_path, create_readme
from LE_LightBoundarySim import sim_eta


def main():
	"""
	NAME
		LE_2DLBS.py
	
	PURPOSE
		Simulate coloured noise trajectories in 2D.
	
	EXECUTION
		
	FLAGS
		-a --alpha		0.1		Slope of the potential
		-X --wallpos	10.0	Position of wall
		-D --Delta		0.0		Width of wall onset in units of X
		-r --nruns		100		Number of runs for each (x0,y0)
		-t --timefac	1.0		Multiply t_max by factor
		-v --verbose	False	Print useful information to screen
		-h --help		False	Print docstring and exit
	
	EXAMPLE
	
	NOTES
	
	BUGS
	
	HISTORY
		20 February 2016	Adapted from LE_LightBoundarySim.py
	"""	
	me = "LE_2DLBS.main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## INPUT OPTIONS
	
	parser = optparse.OptionParser(conflict_handler="resolve")	
	parser.add_option('-a','--alpha',
                  dest="a",default=0.1,type="float",
				  help="The steepness of the potential.")
	parser.add_option('-X','--wallpos',
                  dest="X",default=10.0,type="float")
	parser.add_option('-D','--Delta',
                  dest="Delta",default=0.0,type="float")		
	parser.add_option('-r','--nrun',
                  dest="Nrun",default=100,type="int")
	parser.add_option('--dt',
                  dest="dt",default=0.01,type="float")		
	parser.add_option('-t','--timefac',
                  dest="timefac",default=1.0,type="float")	 
	parser.add_option('-v','--verbose',
                  dest="vb",default=False,action="store_true",
				  help="Print useful information to screen.")	
	parser.add_option('-R','--radius',
                  dest="R",default=1.0,type="float")
	parser.add_option('--schematic',
                  dest="schematic",default=False,action="store_true")
	parser.add_option('-h','--help',
                  dest="help",default=False,action="store_true",
				  help="Print docstring.")					  
	opts, argv = parser.parse_args()
	# opts = vars(opts)
	if opts.help: print main.__doc__; return
	a		= opts.a
	X		= opts.X
	Delta	= opts.Delta
	Nrun	= opts.Nrun
	global dt; dt = opts.dt
	timefac = opts.timefac
	vb		= opts.vb
	R = opts.R
	schematic = opts.schematic
	
	if Delta!=0.0:
		print me+"WARNING: Resetting Delta = 0.0"
		Delta = 0.0
	
	if vb: print "\n==\n"+me+"Input parameters:\n\t",opts

	## ----------------------------------------------------------------
	## SETUP CALCULATIONS
	
	## Simulation time
	tmax = 1e4*timefac
	
	## Space
	xmax = lookup_xmax(X,a)
	xmin = 0.8*X	## Simulation cutoff
	ymax = 0.5
	assert (R>=ymax), me+"The wall must enclose the volume."
			
	## Injection x coordinate
	xini = 0.9*X
		
	## Histogramming; bin edges
	Nxbin = 200
	Nybin = 50
	xbins = calculate_xbin(xini,X,xmax,Nxbin)
	ybins = np.linspace(0.0,ymax,Nybin+1)
	
	## Particles	
	Nparticles = 1*len(ybins)*Nrun

	## Initial noise drawn from Gaussian
	eIC = np.random.normal(0.0,1.0/a,[Nparticles,2])
	
	## Centre of circle for curved boundary
	R2 = R*R
	c = [X-np.sqrt(R2-ymax*ymax),0.0]
	
	## Filename; directory and file existence; readme
	hisdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+"_2D_X"+str(X)+"_R"+str(R)+"_r"+str(Nrun)+"_dt"+str(dt)+"/"
	hisfile = "BHIS_2D_a"+str(a)+"_X"+str(X)+"_R"+str(R)+"_r"+str(Nrun)+"_dt"+str(dt)
	filepath = hisdir+hisfile
	check_path(filepath, vb)
	create_readme(filepath, vb)
		
	## ----------------------------------------------------------------
	## SCHEMATIC IMAGE
	if schematic:
		draw_schematic(xmin,xbins,ybins,c,R,hisdir+"Schematic_"+hisfile+".png",vb)
	exit()
	
	## ----------------------------------------------------------------
	## SIMULATION
	
	if vb: print me+"Computing",Nparticles,"trajectories"
	
	## Precompute exp(-t/a^2)
	expmt = np.exp(-np.arange(0,tmax,dt)/(a*a)) if a>0 else np.array([1.]+[0.]*(int(tmax/dt)-1))
	## Initialise histogram in space
	H = np.zeros((Nxbin,Nybin))
	## Counter for noise initial conditions
	i = 0
	
	## Loop over initial coordinates
	for yini in ybins:
		## Perform several runs
		for run in xrange(Nrun):
			## x, y are coordinates as a function of time
			x, y = boundary_sim((xini,yini), eIC[i], a, X,Delta, xmin,ymax,
				R2,c, tmax,expmt, (run%50==0))
			H += np.histogram2d(x,y,bins=[xbins,ybins],normed=False)[0]
			i += 1
	H = (H.T)[::-1]
	## When normed=False, need to divide by the bin area
	H /= np.outer(np.diff(ybins),np.diff(xbins))
	## Normalise by number of particles
	H /= Nparticles
	save_data(filepath, H, vb)
	
	if vb: print me+"execution time",round(time.time()-t0,2),"seconds"
	
	return filepath
	
## ====================================================================

def boundary_sim(x0y0, exy0, a, X,D, xmin,ymax, R2,c, tmax,expmt, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<xmin.
	Dynamically adds more space to arrays.
	"""
	me = "LE_2DLBS.boundary_sim: "
	
	## Initialisation
	x0,y0 = x0y0
	nstp = int(tmax/dt)
	exstp = nstp/10
	if vb: print me+"a = ",a,"; IC =",x0y0
	
	## Simulate eta
	if vb: t0 = time.time()
	ex = sim_eta(exy0[0], expmt, nstp, a, dt)
	ey = sim_eta(exy0[1], expmt, nstp, a, dt)
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	## Spatial variables
	if vb: t0 = time.time()
	
	## Construct y with periodic boundaries
	y = calculate_y(y0,ey,ymax)
	
	## Iteratively compute x
	x = np.zeros(nstp); x[0] = x0
	i,j = 0,0
	## Euler steps to calculate x(t)
	while x[i] > xmin:
		x[i+1] = x[i] + dt*( force_2D(x[i],y[i],R2,c[0]) + ex[i] )
		i +=1
		## Extend array if necessary
		if i == len(x):
			ex = np.append(ex,sim_eta(ex[-1],expmt[:exstp],exstp))
			x = np.append(x,np.zeros(exstp))
			ey_2 = sim_eta(ey[-1],expmt[:exstp],exstp)
			ey = np.append(ey,ey_2)
			y = np.append(y,calculate_y(y[-1],ey_2,ymax))
			j += 1
	if j>0: print me+"trajectory array extended",j,"times."
	if vb: print me+"Simulation of x",round(time.time()-t0,1),"seconds for",i,"steps"
	
	## Clip trailing zeroes from y and x
	x, y = x[:i], y[:i]	
	return [x,y]
	
## ====================================================================

def calculate_y(y0,ey,ymax):
	"""
	Calculate y trajectory.
	No force -- independent of position.
	Periodic boundary at y = =/- ymax and reflecting at y = 0.0.
	"""
	y = y0 + dt*np.cumsum(ey)
	while (y>ymax).any() or (y<0.0).any():
		idg = (y>ymax)
		y[idg] = 2*ymax - y[idg]
		idl = (y<0.0)
		y[idl] *= -1.0
	return y
	
## ====================================================================

def force_2D(x,y,R2,cx):
	"""
	The force for a curved wall.
	"""
	f = 0.0
	if x*x + y*y > R2 and x > cx:
		f = -1.0
	return f
	
## ====================================================================

def lookup_xmax(X,a):
	"""
	2015/12/08 NEW DEFINITIONS
	Lookup table for xmax
	Limited testing; X=10.0 only, up to a=1.0
	"""
	if   a<=0.08:	xmax=1.4*X
	elif a<=0.1:	xmax=1.15*X
	elif a<=0.2:	xmax=1.1*X
	elif a<=0.3:	xmax=1.02*X
	elif a<=0.4:	xmax=1.005*X
	elif a<=0.5:	xmax=1.003*X
	elif a<=0.6:	xmax=1.003*X
	elif a<=0.7:	xmax=1.003*X
	elif a<=0.8:	xmax=1.002*X
	elif a<=0.9:	xmax=1.002*X
	else:			xmax=1.001*X
	return xmax
	
def calculate_xbin(xinit,X,xmax,Nxbin):
	"""
	Return histogram bins in x
	"""
	# xbins = np.linspace(xinit,xmax,Nxbin+1)
	## Extra bins for detail in wall region
	NxbinL = Nxbin/2; NxbinR = Nxbin - NxbinL
	xbins = np.unique(np.append(np.linspace(xinit,X,NxbinL+1),np.linspace(X,xmax,NxbinR+1)))
	return xbins

## ====================================================================

def draw_schematic(xmin,xbins,ybins,c,R,outfile,vb=False):
	"""
	A schematic of the simulation space
	"""
	me = "LE_2DLBS.draw_schematic: "
	if os.path.isfile(outfile):
		if vb: print me+"Schematic exists. Not overwriting."
		return
	loff = 1.0
	## Get spatial parameters
	xini = xbins[0]
	X	 = xbins[len(xbins)/2]
	xmax = xbins[-1]
	ymax = ybins[-1]
	## Wall region
	plt.axvspan(X,xmax, color="r",alpha=0.05,zorder=0)
	## Wall boundary
	circle = plt.Circle(c,R,facecolor='w',lw=2,edgecolor='r',zorder=1)
	plt.gcf().gca().add_artist(circle)
	## Remove left arc
	plt.axvspan(xmin-loff,X-0.1, color="w",zorder=2)
	# plt.gcf().gca().fill_between(xmin-1.0,X-0.1, 0)
	## Lines
	plt.hlines([-ymax,ymax],0.0,xmax,
		colors='k', linestyles='-', linewidth=2.0,zorder=3)
	plt.vlines([xmin,xini,X],-ymax,ymax,
		colors='k', linestyles=["-","--",":"], linewidth=2.0,zorder=3)
	## Outside simulation
	plt.axvspan(xmin,xmin-loff, color="k",alpha=0.1,zorder=2)
	## Annotations
	plt.annotate("Not simulated",xy=(xmin-0.5*loff,0.0),xycoords="data",
			horizontalalignment='center', verticalalignment='center')
	plt.annotate("Wall region",xy=(0.5*(X+xmax),0.0),xycoords="data",
			horizontalalignment='center', verticalalignment='center')
	plt.annotate("Simulation boundary",xy=(xmin,-0.5*ymax),xycoords="data",
			horizontalalignment='center', verticalalignment='center')
	plt.annotate("Injection line",xy=(xini,+0.5*ymax),xycoords="data",
			horizontalalignment='center', verticalalignment='center')
	plt.annotate("Wall boundary",xy=(X,-0.5*ymax),xycoords="data",
			horizontalalignment='center', verticalalignment='center')
	## Show bins
	# plt.hlines(ybins,xini,xmax, colors='k', linestyles="-",linewidth=0.2,zorder=2.1)
	# plt.vlines(xbins,-ymax,ymax,colors='k', linestyles="-",linewidth=0.2,zorder=2.1)
	## Clip image, name axes, title
	plt.xlim([xmin-loff,xmax])
	plt.ylim([-ymax,ymax])
	plt.xlabel("$x$")
	plt.ylabel("$y$")
	plt.title("Schematic of simulation space")
	## Save and close
	plt.savefig(outfile)
	if vb: print me+"Figure saved to",outfile
	plt.clf()
	return

## ====================================================================
## ====================================================================
if __name__=="__main__": main()