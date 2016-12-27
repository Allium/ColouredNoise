
import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import optparse, os, time
from datetime import datetime

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
	a = 10.0
	timefac = 0.3*a
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
	if 1:#timefac<=0.5:
		try: xy3 = np.vstack([xy1,xy2]); rcoord3 = np.hstack([rcoord1,rcoord2])
		except: xy3 = xy1; rcoord3 = rcoord1
		plot_traj(xy3,rcoord3,R,S,lam,nu,force_dlin,a,dt,True)

	## HISTOGRAM
	if 0:#timefac>=0.5:
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
	
##=============================================================================
	
from platform import system

def plot_traj(xy,rcoord,R,S,lam,nu,force_dx,a,dt,vb):
	"""
	"""
	me = "LE_RunPlot.plot_traj: "

	fig = plt.figure(); ax = fig.gca()
	cm = plt.get_cmap("winter"); CH = 100; NPOINTS = xy.shape[0]/CH
	if system() == "Linux":
		ax.set_prop_cycle("color",[cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
	else:
		ax.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
	
	for i in range(NPOINTS-1):
		ax.plot(xy[i*CH:(i+1)*CH+1,0],xy[i*CH:(i+1)*CH+1,1],zorder=1)
	
	## Plot force arrows
	fstr = ""
	if 0:
		fstr = "F"
		numarrow = 100	## not number of arrows plotted
		for j in range(0, xy.shape[0], xy.shape[0]/numarrow):
			uforce = force_dx(xy[j],rcoord[j],R,S)
			if uforce.any()>0.0:
				plt.quiver(xy[j,0],xy[j,1], uforce[0],uforce[1],
						width=0.008, scale=20.0, color="purple", headwidth=2, headlength=2, zorder=2)
		print me+"Warning: angles look off."
							
	## Plot walls
	ang = np.linspace(0.0,2*np.pi,360)
	ax.plot(R*np.cos(ang),R*np.sin(ang),"y--",(R+lam)*np.cos(ang),(R+lam)*np.sin(ang),"y-",lw=2.0, zorder=3)
	ax.plot(S*np.cos(ang),S*np.sin(ang),"r--",(S-lam)*np.cos(ang),(S-lam)*np.sin(ang),"r-",lw=2.0, zorder=3)
	
	# ax.set_xlim((-R-lam-0.1,R+lam+0.1));	ax.set_ylim((-R-lam-0.1,R+lam+0.1))
	limmax = np.max([np.abs(ax.get_xlim()),np.abs(ax.get_ylim())])
	ax.set_xlim(-limmax,limmax)
	ax.set_ylim(-limmax,limmax)
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$y$")
	ax.grid()
	
	## Save
	if not nosave:
		plotfile = outdir+str(dt)+\
					"/TRAJ"+fstr+"_CIR_DN_a"+str(a)+"_R"+str(R)+"_S"+str(S)+\
					"_l"+str(lam)+"_n"+str(nu)+"_t"+str(round(rcoord.size*dt/5e2,1))+"_dt"+str(dt)+".png"
		fig.savefig(plotfile)
		if vb:	print me+"TRAJ figure saved to",plotfile
	if vb:	plt.show()
	plt.close()
	
	return
	
## ====================================================================
## ====================================================================
if __name__=="__main__":
	main()
