
import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import optparse, os, time
from datetime import datetime

from LE_LightBoundarySim import sim_eta
from LE_SSim import force_lin, force_dlin, eul
from LE_RunPlot import plot_traj
from LE_SPressure import plot_wall

"""
See what happens to a particle released under controlled conditions.
"""

## ====================================================================


def main():
	a = 1.0
	timefac = 0.02
	dt = 0.01
	R = 2.0
	S = R
	lam = 0.0
	global nu; nu = 1.0
	vb = True
	sn = +1.0
	ss = 10001

	fxy = lambda xy, r: nu*force_dlin(xy,r,R,S)
		
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
	np.random.seed(12)
	if a == 0.0:	eIC = np.sqrt(2/dt)*np.random.normal(0.0, 1.0, 2)
	else:			eIC = 1./np.sqrt(a)*np.random.normal(0.0, 1.0, 2)
		
	## Integration algorithm
	xy_step = lambda xy, r2, exy: eul(xy, r2, fxy, exy, dt)
							
	## ----------------------------------------------------------------
	## SIMULATION

	## Precompute exp(-t)
	expmt = np.exp((np.arange(-10*a,dt,dt))/a)

	np.random.seed(ss)
	xy1, rcoord1 = boundary_sim([rini,0.0], +sn, eIC, a, xy_step, dt, tmax, expmt)
	# np.random.seed(ss)
	# xy2, rcoord2 = boundary_sim([rini,0.0], -sn, eIC, a, xy_step, dt, tmax, expmt)

	## TRAJECTORY
	if timefac<=0.5:
		try: xy3 = np.vstack([xy1,xy2]); rcoord3 = np.hstack([rcoord1,rcoord2])
		except: xy3 = xy1; rcoord3 = rcoord1
		plot_traj(xy3,rcoord3,R,S,lam,nu,force_dlin,a,dt,True)

	## HISTOGRAM
	if timefac>=0.5:
		h1, bins1 = np.histogram(rcoord1, bins=100, normed=True)[0:2]
		# h2, bins2 = np.histogram(rcoord2, bins=100, normed=True)[0:2]
		plt.plot(bins1[1:], h1/bins1[1:])
		# plt.plot(bins2[1:], h2/bins2[1:])
		plt.xlabel("$r$")
		plt.grid()
		plot_wall(plt.gca(), "dlin", [R,S,0,nu], bins1)
		plt.show()

	return
	
##=============================================================================

def boundary_sim(xyini, sn, exyini, a, xy_step, dt, tmax, expmt):

	## Initialisation
	x0,y0 = xyini
	nstp = int(tmax/dt)
	
	## Simulate eta
	exy = np.vstack([sn*sim_eta(exyini[0], expmt, nstp, a, dt), sim_eta(exyini[1], expmt, nstp, a, dt)]).T
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
