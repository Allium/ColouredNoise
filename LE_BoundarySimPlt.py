import numpy as np
import matplotlib.pyplot as plt
import os, time
import optparse
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import fftconvolve
from sys import argv

from LE_Utils import *


def main():
	"""
	NAME
	LE_BoundatSimPlt.py
	
	PURPOSE
	Simulate and plot trajectories close to the wall (at +X) in the BW setup
	
	EXECUTION
	python LE_BoundarySimPlt.py -b 0.1 -X 1.0
	
	EXAMPLE
	
	STARTED
	04 May 2015
	
	NOTES
	-- There is an option to save a trajectory file. For this, use say 2 runs per y0 and skip every other y0.
	
	BUGS
	-- xmax is poorly hard-coded
	-- Setting dt in expmt
	
	
	LOGIC
	Read in options
	Simulation parameters
	Histogram / output parameters	
	Loop over y0:
		Simulate multiple trajectories	
			Make histogram of each trajectory
			Weight trajectory based on yi, yf			
	Histogram all trajectories
	Save histogram file	
	Plot 2D pdf
	Project 1D pdfs	
	Calculate currents
	Stream-/quiverplot
	"""	
	
	me = "LE_BoundarySimPlt.main: "
	t0 = time.time()
	
	## ----------------------------------------------------------------
	## Input options
	
	parser = optparse.OptionParser()	
	parser.add_option('-b',
                  dest="potmag",
                  default=0.1,
                  type="float",
                  )
	parser.add_option('-X','--wallpos',
                  dest="wall",
                  default=1.0,
                  type="float",
                  )
	parser.add_option('-r','--nrun',
                  dest="Nrun",
                  default=5000,
                  type="int",
                  )		
	parser.add_option('-t','--timefac',
                  dest="timefac",
                  default=1.0,
                  type="float",
                  )				  
	parser.add_option('-s','--smooth',
                  dest="smooth",
                  default=0.0,
                  type="float",
                  )			  
	parser.add_option('-v','--verb',
                  dest="verb",
                  default=True,
                  action="store_false",
                  )					  
	opt = parser.parse_args()[0]
	b		= opt.potmag
	X		= opt.wall
	Nrun	= opt.Nrun
	timefac = opt.timefac
	smooth	= opt.smooth
	vb		= opt.verb

	print "\n==  "+me+"b =",b," X =",X," Nruns =",Nrun," ==\n"

	## ----------------------------------------------------------------
	## Parameters
	
	## Simulation	
	x0 = 0.9*X
	tmax = 1e3*timefac	## This is large enough to ensure re-hitting?
	global dt; dt = 0.02
	
	## Histogramming
	xmax = 1.1*X	## Should depend on b
	xmin = 0.8*X	## For plotting, histogram bins and simulation cutoff
	ymax = round(2.0*np.sqrt(b),1)		## Hard-coded!
	Nbin = 50
	xbins = np.linspace(xmin,xmax,Nbin+1)
	ybins = np.linspace(-ymax,ymax,Nbin+1)
	xbc, ybc = 0.5*(xbins[1:]+xbins[:-1]), 0.5*(ybins[1:]+ybins[:-1])
	dx = (xmax-x0)/Nbin; dy = 2*ymax/Nbin
	
	hisfile, trafile, pdffile, strfile = boundaryfilenames(b,X,ymax,Nbin,Nrun)
	saveplot = True
	trajplot = False
	savedata = True
	fs = 25
	
	## ----------------------------------------------------------------
	## Load histogram; or simulate trajectory and build histogram
	
	try:
		H = np.load(hisfile)
		print me+"Histogram data found:",hisfile
		
	except IOError:
		print me+"No histogram data found. Simulating trajectories..."
		t0 = time.time()
		## Precompute exp(-t) and initialise histogram
		expmt = np.exp(-np.arange(0,tmax,dt))
		H = np.zeros((Nbin,Nbin))
		## Loop over initial y-position
		for y0 in ybins:
			if vb: print me+str(Nrun),"trajectories with y0 =",round(y0,2)
			for run in xrange(Nrun):
				x, y = boundary_sim((x0,y0), b, X, FBW, xmin, tmax, expmt, False)
				if trajplot: plt.plot(x-1,y)	## NOTE -1 to make wall offset
				h, xbins, ybins = np.histogram2d(x,y,bins=[xbins,ybins])
				H += h*histogram_weight(y0,y[-1],b)
		H = (H.T)[::-1]
		## Save
		if trajplot:
			# plot_walls(plt,X,xmax,ymax,1);plt.xlim([xmin,1.4]);plt.ylim([-ymax,ymax]);\
				# plt.xlabel("$x$",fontsize=fs);plt.ylabel("$\eta$",fontsize=fs);\
				# plt.savefiplot_walls(plt,X,xmax,ymax,1);plt.xlim([xmin,1.4]);plt.ylim([-ymax,ymax]);\
			plot_walls(plt,X-1,x,xmax,ymax,1)
			plt.xlim([X-1,0.8]);plt.ylim([-0.1,1.0])
			plt.xlabel("$\Delta x$",fontsize=fs);plt.ylabel("$\eta$",fontsize=fs)
			plt.savefig(trafile,bbox_inches='tight')
			if vb: print me+"trajectory plot saved to",trafile
			plt.show()
		if savedata:	save_data(hisfile, H, True)
	
	## Normalise
	H /= np.trapz(np.trapz(H,dx=dx,axis=1), dx=dy)

	## ----------------------------------------------------------------
	## Plot PDFs
	xminplot = x0-0.25*(x0-xmin)
	
	## 1D
	if 0:
		t1 = time.time()
		for i in xrange(Nbin):
			fig, (ax1,ax2) = plt.subplots(2)
			ax1.plot(ybc,H[:,i])
			ax1.set_xlabel("$\eta$",fontsize=fs);	ax1.set_ylabel("$p(\eta|x="+str(round(x0+i*dx,3))+")$",fontsize=fs)
			ax2.plot(xbc,H[i,:])
			ax2.set_xlabel("$x$",fontsize=fs);	ax2.set_ylabel("$p(x|\eta="+str(round(ymax-i*dy,2))+")$",fontsize=fs)
			fig.tight_layout()
			pdffile1 = os.path.split(pdffile)[0]+"/1DPDF/"+os.path.split(pdffile)[1][:-4]+"_"+str(i)+".png"
			if saveplot:
				fig.savefig(pdffile1)
			plt.close()
		if saveplot and vb:
			pdffile1 = os.path.split(pdffile)[0]+"/1DPDF/"+os.path.split(pdffile)[1][:-4]+"_*.png"
			print me+"1D PDF plots saved to",pdffile1,". Time",round(time.time()-t1,2),"seconds."
		return
	
	## 2D
	plt.imshow( np.log10(H+1) ,extent=[xmin,xmax,-ymax,ymax],aspect="auto")
	# plt.imshow( H ,extent=[xmin,xmax,-ymax,ymax],aspect="auto")
	plot_walls(plt, X, xmax, ymax, 1)
	# plt.xlim([xminplot,xmax]);	plt.ylim([-ymax,ymax])
	plt.xlim([0.9,xmax]);	plt.ylim([-0.4,0.4])
	plt.xlabel("$x$",fontsize=fs);	plt.ylabel("$\eta$",fontsize=fs)
	cbar = plt.colorbar(orientation="horizontal"); cbar.set_label("$\log\, p$", fontsize=fs-7)
	plt.tight_layout()
	if saveplot:
		plt.savefig(pdffile,facecolor="w",bbox_inches='tight')
		if vb: print me+"PDF plot saved to",pdffile
	plt.clf()
	exit()

	## ----------------------------------------------------------------
	## Calculate currents
	
	gx, gy = np.meshgrid(xbc, ybc); gy = -gy
	Jx, Jy = J_BW(H,b,FBW(gx,b,X),gy)
	
	## Smooth data
	if smooth is not 0.0:
		Jx = gaussian_filter(Jx, smooth)
		Jy = gaussian_filter(Jy, smooth)
	
	## ----------------------------------------------------------------
	## Plot current
		
	plt.quiver(gx,gy,Jx,Jy)
	plot_walls(plt, X, xmax, ymax, 1)
	plt.xlim([xminplot,xmax]);	plt.ylim([-ymax,ymax])
	plt.xlabel("$x$",fontsize=fs);	plt.ylabel("$\eta$",fontsize=fs)
	if saveplot:
		plt.savefig(strfile, facecolor="w")
		if vb: print me+"STR plot saved to",strfile
	plt.close()

	
	print me+"Total time",round(time.time()-t0,1),"seconds"	
	
	return
	
## ====================================================================
def boundary_sim(x0y0, b, X, f, xmin, tmax, expmt=None, vb=False):
	"""
	Run the LE simulation from (x0,y0), stopping if x<x0
	
	NOTES
	-- Assuming tmax is sufficient time to explore and return.
	If this turns out to not be the case, will have to dynamically increase array size.
	-- Computing eta and then x. Maybe faster to have one loop.
	"""

	me = "LE_BoundarySimPlt.boundary_sim: "
	
	## Initialisation
	x0,y0 = x0y0
	seed = np.random.randint(1000000); np.random.seed(seed)
	nstp = int(tmax/dt)
	xi = np.random.normal(0, 1, nstp)
	
	t0 = time.time()
	## OU noise
	if expmt is None or expmt.shape[0]!=nstp: expmt = np.exp(-np.linspace(0,tmax,nstp))
	y = y0*expmt + np.sqrt(2*b)*dt * fftconvolve(expmt,np.append(np.zeros(nstp),xi),"full")[nstp-1:-nstp]
	if vb: print me+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps"
	
	t0 = time.time()
	## Variable of interest
	x = np.zeros(nstp); x[0] = x0
	## Euler steps to calculate x(t)
	for i in xrange(1,nstp):
		x[i] += x[i-1] + dt*(f(x[i-1],b,X) + y[i-1])
		if x[i] < xmin:	break
	if i==nstp-1: print me+"Trarray ran out of space. BUG!"
	if vb: print me+"Simulation of x  ",round(time.time()-t0,1),"seconds for",len(x),"steps"
	
	## Clip y and x to be their actual lengths
	x, y = x[:i+1], y[:i+1]

	return np.vstack([x,y])

## ====================================================================

def histogram_weight(yi,yf,var):
	"""
	Weights depend on starting position and finishing position.
	Probability of y obeys Gaussian with zero mean and variance ???
	"""
	return np.exp(-(yi*yi+yf*yf)/(2.0*var))

	
## ====================================================================
## ====================================================================
if __name__=="__main__": main()