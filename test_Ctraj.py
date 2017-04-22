me0 = "test_Ctraj"

import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator
import optparse, os, time
from datetime import datetime
from platform import system
import os

from LE_LightBoundarySim import sim_eta
from LE_CSim import force_dlin, eul
from LE_Utils import fs, set_mplrc

set_mplrc(fs)

"""
Plot trajectory
"""

## ====================================================================

def main():
	"""
	"""
	me = me0+".main: "
	
	a = 1.0
	timefac = 1.0
	dt = 0.01
	R = 2.0
	S = 0.0
	vb = True

	fx = lambda x: force_dlin([x,0],R,S)[0]

	outdir = "./Pressure/TRAJ_CAR_DL_a%.1f_R%.1f_S%.1f_t%.1f"%(a,R,S,timefac)
	if not os.path.isdir(outdir):
		os.mkdir(outdir)
		print me+"Created directory",outdir
		
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS

	## Simulation time
	tmax = 5e2*timefac

	## Simulation limits
	xmax = R+3.0
	xmin = S-3.0
	## Injection x coordinate
	xini = 0.5*(S+R)

	## Initial noise drawn from Gaussian
	if a == 0.0:	eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0)
	else:			eIC = dt/np.sqrt(a)*np.random.normal(0.0, 1.0)
		
	## Integration algorithm
	x_step = lambda x, ex: eul(x, ex, fx, dt)
							
	## ----------------------------------------------------------------
	## SIMULATION

	## Precompute exp(-t)
	expmt = np.exp((np.arange(-10*a,dt,dt))/a)

	x, ex = boundary_sim(xini, eIC, a, x_step, dt, tmax, expmt)
	
	## Coarsen to speed up plotting
	crsn = 5
	if crsn>1:
		x = x[::crsn]
		ex = ex[::crsn]
	
	## ----------------------------------------------------------------
	## PLOT TRAJECTORY
	
	## Number of steps per frame
	nsteps = int(tmax/dt/crsn)
	stride = 100/crsn	## Make a frame every stride timesteps
	numf = nsteps/stride
	
	## Loop over frames
	for fnum in range(numf):
		ti = time.time()
		fig = plot_traj(x[:fnum*stride],ex[:fnum*stride],R,S,a,dt,x.shape[0],True)
	
		## Save
		plotfile = outdir+"/f%04d.png"%(fnum)
		fig.savefig(plotfile)
		plt.close()
		if not fnum%10: print me+"Frame %04d/%04d saved. Time per file %.1f seconds."%(fnum,numf,time.time()-ti)

	os.system("ffmpeg -r 10 -f image2 -s 1920x1080 -i f%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p TRAJ_CAR_DL_a%.1f_R%.1f_S%.1f_t%.1f.mp4"%(a,R,S,t))
	
	return
	
##=============================================================================
	
def plot_traj(x,ex,R,S,a,dt,totpts,vb):
	"""
	"""
	me = "test_traj.plot_traj: "

	fig = plt.figure(); ax = fig.gca()
	cm = plt.get_cmap("GnBu")
	
	## ------------------------------------------------------------------------
	## LINE COLOUR
	## Constant colour for each line segment of CH points
	## Colour decays away and hits min value.
	
	cm = plt.get_cmap("GnBu")
	
	CH = int(a/dt/20)	## Points in a colouring chunk
	NCH = totpts/CH ## Total number of chunks
	NCHC = 50 ## Number of coloured chunks, after which line is cmin
	cmin = 0.3
	
	## colourlist is linearly declining until NCHC chunks have been coloured, then constant colour
	colourlist = [cm(max(1.*(NCHC-i)/(NCHC-1),cmin)) for i in range(NCHC)] + [cm(cmin)]*(NCH-NCHC)
	# colourlist = colourlist[::-1]
	
	ax.set_prop_cycle("color", colourlist)
	## ------------------------------------------------------------------------	

	## Plot segments
	x=x[::-1]; ex=ex[::-1]
	for i in range(x.shape[0]/CH):
		ax.plot(x[i*CH:(i+1)*CH+1],ex[i*CH:(i+1)*CH+1],zorder=1)
		
	## Plot wall
	xfine = np.linspace(S-3,R+3,100*(R-S+6)+1)
	ax.plot(xfine, -force_dlin([xfine,0],R,S)[0], "k-")
	
	ax.set_xlim(S-2,R+2)
	ax.set_ylim(-3,+3)
	ax.xaxis.set_major_locator(NullLocator())
	ax.yaxis.set_major_locator(NullLocator())
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	ax.grid()
	
	return fig
	
##=============================================================================

def boundary_sim(x0, ex0, a, x_step, dt, tmax, expmt):

	nstp = int(tmax/dt)
	
	## Simulate eta
	ex = sim_eta(ex0, expmt, nstp, a, dt)
	x = np.zeros(nstp); x[0] = x0
	
	## Calculate trajectory
	for i in xrange(0,nstp-1):
		x[i+1] = x[i] + x_step(x[i],ex[i])
		
	return x, ex
	
## ====================================================================
## ====================================================================
if __name__=="__main__":
	main()

"""
ffmpeg -r 10 -f image2 -s 1920x1080 -i f%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p TRAJ_CAR_DL_a1.0_R2.0_S2.0_t3.0.mp4
"""
