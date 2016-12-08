import os, time
import timeit
import numpy as np
import scipy as sp
import scipy.integrate
from matplotlib import pyplot as plt
from LE_Utils import fs, set_mplrc

from LE_SSim import force_dlin
from LE_CSim import force_clin, force_mlin, force_nlin, force_dlin, force_ulin

# set_mplrc(fs)

## 1D force
if 0:
	R = 2.0
	S = 0.0
	T = 0.0
		
	x = np.linspace(-R-4,R+4,1000)	## For nlin, mlin
	x = np.linspace(-S-4,R+4,1000)	## For dlin

	#fx = force_dlin([x,0],R,S,T)[0]	## For x
	fx = force_dlin([x,x],x,R,S)[0]	## For r

	U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()

	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])

	ax.plot(x, U, "-", label=r"$U(r)$", lw=3)
	ax.plot(x, fx, "-", label=r"$f(r)$", lw=3)

	ax.set_xlim((x[0],x[-1]))
	ax.set_xlabel(r"$r$", fontsize=fs["fsa"])
	#ax.set_ylabel(r"$f,U$", fontsize=fs["fsa"])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.grid()
	ax.legend(loc="best",fontsize=fs["fsl"]).get_frame().set_alpha(0.5)

	plt.show()
	exit()
	
## 2D force
elif 1:
	R = 2.0
	A = 0.5
	l = 1.0
		
	x = np.linspace(-R-1*A,R+1*A,201)
	y = np.linspace(0,2*l,101)

	fxy = np.array([force_ulin([xi,yi],R,A,l) for xi in x for yi in y]).reshape((x.size,y.size,2))
	fxy = np.rollaxis(fxy,2,0)
	f = np.sqrt((fxy*fxy).sum(axis=0))

	### This isn't quite right
	U = -sp.integrate.cumtrapz(fxy[0],x,axis=0,initial=0.0) -sp.integrate.cumtrapz(fxy[1],y,axis=1,initial=0.0)
	U -= U.min()
	
	##-------------------------------------------------------------------------
	## Plot potential
	
	# fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	# fig.canvas.set_window_title("Potential")
	
	# ax.contourf(x,y,U.T)
	
	# ## Wall boundary
	# yfine = np.linspace(y[0],y[-1],1000)
	# ax.scatter(R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)
	# ax.scatter(0+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)

	# ax.set_xlim((x[0],x[-1]))
	# ax.set_ylim((y[0],y[-1]))
	# ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	# ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	
	##-------------------------------------------------------------------------
	## Plot force
	
	fig, axs = plt.subplots(1,2, figsize=fs["figsize"], sharey=True)
	fig.canvas.set_window_title("Force")

	ax = axs[0]
	
	ax.contourf(x,y,f.T)
	
	# U, V = fxy
	# stp = (2,2)
	# ax.quiver(X[::stp[0],::stp[1]], Y[::stp[0],::stp[1]], U[::stp[0],::stp[1]], V[::stp[0],::stp[1]], units="width")
	
	## Wall boundary
	yfine = np.linspace(y[0],y[-1],1000)
	ax.scatter(R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)
	ax.scatter(-R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)

	ax.set_xlim((x[0],x[-1]))
	ax.set_ylim((y[0],y[-1]))
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	ax.set_title(r"force_ulin")
	
	
	## Check prediction 
	ax = axs[1]
	
	X, Y = np.meshgrid(x, y)
	fcalc = +(X-R-A*np.sin(2*np.pi*Y/l))*np.sqrt(1+(2*np.pi/l)**2*A*np.cos(2*np.pi*Y/l)**2)*(X>(R+A*np.sin(2*np.pi*Y/l))) +\
			-(X+R-A*np.sin(2*np.pi*Y/l))*np.sqrt(1+(2*np.pi/l)**2*A*np.cos(2*np.pi*Y/l)**2)*(X<(-R+A*np.sin(2*np.pi*Y/l)))
	ax.contourf(x,y,fcalc)
	
	## Wall boundary
	yfine = np.linspace(y[0],y[-1],1000)
	ax.scatter(R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)
	ax.scatter(-R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)

	ax.set_xlim((x[0],x[-1]))
	ax.set_ylim((y[0],y[-1]))
	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	ax.set_title(r"calculation")

	##-------------------------------------------------------------------------

	plt.show()
	
	exit()







"""

N = 10000
xy = np.linspace(0.0,1.0,10000).reshape((2,5000))
T1, T2, T3 = np.zeros((3,N))
for i in range(N):
	t0 = time.time()
	np.array([force_dlin(xyi,np.sqrt((xyi*xyi).sum()),R,S) for xyi in xy])[:,0]
	T1[i] = time.time()-t0
for i in range(N):
	t0 = time.time()
	np.array([force_dlin2(xyi,np.sqrt((xyi*xyi).sum()),R,S) for xyi in xy])[:,0]
	T2[i] = time.time()-t0
print "1 %.3g"%(T1.mean())
print "2 %.3g"%(T2.mean())
	
	
exit()
"""
