me0 = "test_traj"

import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import optparse, os, time
from datetime import datetime
from platform import system

from LE_LightBoundarySim import sim_eta
from LE_SSim import force_lin, force_dlin, eul
from LE_SPressure import plot_wall
from LE_Utils import fs, set_mplrc

set_mplrc(fs)

"""
See what happens to a particle released under controlled conditions.
"""

## ====================================================================
nosave = True

def main():
	"""
	"""
	me = me0+".main: "
	
	a = 1.0
	timefac = 0.3*a
	dt = 0.01
	R = 2.0
	S = R
	vb = True
	ss = 10001

	fxy = lambda xy, r: force_dlin(xy,r,R,S)
	
	outdir = "./Pressure/TRAJ_POL_DL_a%.1f_R%.1f_S%.1f_t%.1f"%(a,R,S,timefac)
	if not os.path.isdir(outdir): os.mkdir(outdir)
		
	## ----------------------------------------------------------------
	## SET UP CALCULATIONS

	## Simulation time
	tmax = 5e2*timefac

	## Simulation limits
	rmax = R+3.0
	rmin = 0.0
	## Injection x coordinate
	rini = 0.5*(S+R)

	## Initial noise drawn from Gaussian
#	np.random.seed(12)
	if a == 0.0:	eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0, 2)
	else:			eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0, 2)
		
	## Integration algorithm
	xy_step = lambda xy, r2, exy: eul(xy, r2, fxy, exy, dt)
							
	## ----------------------------------------------------------------
	## SIMULATION

	## Precompute exp(-t)
	expmt = np.exp((np.arange(-10*a,dt,dt))/a)

	np.random.seed(ss)
	xy, rcoord = boundary_sim([rini,0.0], eIC, a, xy_step, dt, tmax, expmt)
	
	## Coarsen
#	xy = xy[::4]
#	rcoord = rcoord[::4]

	## ----------------------------------------------------------------
	## PLOT TRAJECTORY
	
	## Number of steps per frame
	nsteps = int(tmax/dt)
	stride = 100	## Make a frame every stride timesteps
	numf = nsteps/stride
	
	## Loop over frames
	for fnum in range(numf):
		fig = plot_traj(xy[:fnum*stride],rcoord[:fnum*stride],R,S,a,dt,rcoord.size,True)
	
		## Save
		plotfile = outdir+"/f%04d.png"%(fnum)
		fig.savefig(plotfile)
		if not fnum%10: print me+"Frame %04d/%04d saved."%(fnum,numf)
		plt.close()

	return
	
##=============================================================================
	
def plot_traj(xy,rcoord,R,S,a,dt,totpts,vb):
	"""
	"""
	me = "test_traj.plot_traj: "

	fig = plt.figure(); ax = fig.gca()
	cm = plt.get_cmap("GnBu")#winter
	
	## Constant number of colours -- get stretched out
	if 0:
		CH = 100	## Points in a colouring chunk
		NCH = max(xy.shape[0]/CH,1)
		if system() == "Linux":
			colourlist = [cm(1.*i/(NCH)) for i in range(NCH)]	## Constant number of colours
			ax.set_prop_cycle("color", colourlist)
		else:
			ax.set_color_cycle([cm(1.*i/(NCH-1)) for i in range(NCH-1)])
	
		for i in range(NCH):
			seg = xy[i*CH:(i+1)*CH+1]
			ax.plot(seg[:,0],seg[:,1],zorder=1)
			
	## Constant colour-segment length
	else:
		xy = xy[::-1]
		rcoord = rcoord[::-1]
		CH = 100#int(a/dt)	## Points in a colouring chunk
		NCH = totpts/CH ## Number of chunks
		colourlist = [cm(1.*i/(NCH-1)) for i in range(NCH)][::-1]
		ax.set_prop_cycle("color", colourlist) if system()=="Linux" else ax.set_color_cycle(colourlist)
	
		for i in range(rcoord.size/CH):
			seg = xy[i*CH:(i+1)*CH+1]
			ax.plot(seg[:,0],seg[:,1],zorder=1)
							
	## Plot walls
	ang = np.linspace(0.0,2*np.pi,360)
	ax.plot(R*np.cos(ang),R*np.sin(ang),"r-", zorder=3)
	ax.plot(S*np.cos(ang),S*np.sin(ang),"g-", zorder=3)
	
	limmax = R+2.0
	ax.set_xlim(-limmax,limmax)
	ax.set_ylim(-limmax,limmax)
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")
	ax.grid()
	
	return fig
	
##=============================================================================

def boundary_sim(xyini, exyini, a, xy_step, dt, tmax, expmt):

	## Initialisation
	x0,y0 = xyini
	nstp = int(tmax/dt)
	
	## Simulate eta
	exy = np.vstack([sim_eta(exyini[0], expmt, nstp, a, dt), sim_eta(exyini[1], expmt, nstp, a, dt)]).T
	xy = np.zeros([nstp,2]); xy[0] = [x0,y0]
	
	## Calculate trajectory
	for i in xrange(0,nstp-1):
		r = np.sqrt((xy[i]*xy[i]).sum())
		xy[i+1] = xy[i] + xy_step(xy[i],r,exy[i])
			
	rcoord = np.sqrt((xy*xy).sum(axis=1))
	return xy, rcoord
	
## ====================================================================
## ====================================================================
if __name__=="__main__":
	main()

"""
ffmpeg -r 10 -f image2 -s 1920x1080 -i f%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
"""
