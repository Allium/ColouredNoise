import os, time
import timeit
import numpy as np
import scipy as sp
import scipy.integrate
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from mpl_toolkits.mplot3d import axes3d
from LE_Utils import fs, set_mplrc

from LE_CSim import force_clin, force_mlin, force_nlin, force_dlin, force_ulin
from LE_SSim import force_dlin

set_mplrc(fs)

##=============================================================================

## 3D potential -- POLAR
def plot_U3D_polar(ax, R, S):
	"""
	Plot polar potential in 3D. Must be passed axes and parameters.
	"""

	## Create supporting points in polar coordinates
	r = np.linspace(0, R+2, 100)
	p = np.linspace(0, 2*np.pi, 50)
	rr, pp = np.meshgrid(r, p)
	## Transform to Cartesian
	X, Y = rr*np.cos(pp), rr*np.sin(pp)

	## Zero bulk
	if R==S:
		U = 0.5*(rr - R)**2
	## Finite bulk
	else:
		U = 0.5*(rr-R)**2*(rr>=R) + 0.5*(rr-S)**2*(rr<=S)
	
	## Plot walls -- too strong but I've tried
	ax.plot(R*np.cos(pp),R*np.sin(pp),0.0, "r-", alpha=0.2, zorder=1)
	if R!=S:
		ax.plot(S*np.cos(pp),S*np.sin(pp),0.0, "g-", alpha=0.2, zorder=1)
	
	## Plot U
	ax.plot_surface(X, Y, U, rstride=1, cstride=1, alpha=0.15, zorder=2)
	
	## Modify axes
#	ax.set_zlim3d(0, 1)
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$y$')
	ax.set_zlabel(r'$U$')
	ax.xaxis.set_major_locator(NullLocator())
	ax.yaxis.set_major_locator(NullLocator())
	ax.zaxis.set_major_locator(NullLocator())
	
	return
##=============================================================================

if __name__=="__main__":

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
	
	##=============================================================================


	## 3D force CARTESIAN
	if 0:
		R = 4.0
		S = 2.0
		T = 0.0
		
		x = np.linspace(-R-3,R+3,500)	## For nlin, mlin, clin
	#	x = np.linspace(S-4,R+4,500)	## For dlin
		y = np.linspace(0,1,50)

		X, Y = np.meshgrid(x, y, indexing="ij")

		fx = force_clin([x,y],R,S,T)[0]
		U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()
		U = np.meshgrid(U, y, indexing="ij")[0]

		## Plotting
		fig = plt.figure()
		ax = fig.gca(projection="3d")

		## 3D contour plot
		ax.plot_surface(X, Y, U, alpha=0.2, rstride=3, cstride=3, antialiased=True)
	
		## Angle
		ax.elev = 15
		ax.azim = 75
	
		ax.set_xlabel(r"$x$")
		ax.set_ylabel(r"$y$")
		ax.set_zlabel(r"$U$")
	
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
		ax.zaxis.set_major_locator(NullLocator())

		fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
	
		plt.show()
		exit()



	
	##=============================================================================
	
	## 3D POLAR
	if 1:
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		R, S = 2.0, 2.0
		# R, S = 3.0, 1.0
	
		plot_U3D_polar(ax, R, S)
	
		fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
	
		plt.show()
		exit()

	
	##=============================================================================
	
	## 2D force -- for ulin
	if 0:
		R = 1.0
		A = 0.5
		l = 1.0
		
		x = np.linspace(-R-1*A,R+2*A,201)
		y = np.linspace(0,2*l,101)

		fxy = np.array([force_ulin([xi,yi],R,A,l) for xi in x for yi in y]).reshape((x.size,y.size,2))
		fxy = np.rollaxis(fxy,2,0)
		f = np.sqrt((fxy*fxy).sum(axis=0))
	
		X, Y = np.meshgrid(x, y, indexing="ij")
		U = 0.5*(X-R-A*np.sin(2*np.pi*Y/l))**2 * (X>R+A*np.sin(2*np.pi*Y/l)) +\
			0.5*(X+R-A*np.sin(2*np.pi*Y/l))**2 * (X<-R+A*np.sin(2*np.pi*Y/l))
		
		##-------------------------------------------------------------------------
		## Plotting
		lvls = 15
		plt.rcParams["image.cmap"] = "coolwarm"
	
		##-------------------------------------------------------------------------
		## Plot potential
	
		fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
		fig.canvas.set_window_title("Potential")
	
		ax.contourf(x,y,U.T, lvls)

		## Wall boundary
		yfine = np.linspace(y[0],y[-1],1000)
		ax.scatter(R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)
		ax.scatter(-R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)

		ax.set_xlim((x[0],x[-1]))
		ax.set_ylim((y[0],y[-1]))
		ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
		ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
		
		##-------------------------------------------------------------------------
		## Plot force
	
		fig, axs = plt.subplots(1,1, figsize=fs["figsize"], sharey=True)
		fig.canvas.set_window_title("Force")

		ax = axs#[0]
	
		ax.contourf(x,y,f.T, lvls)
	
	#	U, V = fxy
	#	stp = (3,3)
	#	ax.quiver(X[::stp[0],::stp[1]], Y[::stp[0],::stp[1]], U[::stp[0],::stp[1]], V[::stp[0],::stp[1]],
	#			units="width")#, color='r', linewidths=(1,), edgecolors=('k'), headaxislength=5)
	
		## Wall boundary
		yfine = np.linspace(y[0],y[-1],1000)
		ax.scatter(R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)
		ax.scatter(-R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)

		ax.set_xlim((x[0],x[-1]))
		ax.set_ylim((y[0],y[-1]))
		ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
		ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	#	ax.set_title(r"force_ulin")
	
	
	#	## Check prediction 
	#	ax = axs[1]
	#	
	#	X, Y = np.meshgrid(x, y)
	#	fcalc = +(X-R-A*np.sin(2*np.pi*Y/l))*np.sqrt(1+(2*np.pi/l)**2*A*np.cos(2*np.pi*Y/l)**2)*(X>(R+A*np.sin(2*np.pi*Y/l))) +\
	#			-(X+R-A*np.sin(2*np.pi*Y/l))*np.sqrt(1+(2*np.pi/l)**2*A*np.cos(2*np.pi*Y/l)**2)*(X<(-R+A*np.sin(2*np.pi*Y/l)))
	#	ax.contourf(x,y,fcalc)
	
	#	## Wall boundary
	#	yfine = np.linspace(y[0],y[-1],1000)
	#	ax.scatter(R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)
	#	ax.scatter(-R+A*np.sin(2*np.pi*yfine/l), yfine, c="k", s=1)

	#	ax.set_xlim((x[0],x[-1]))
	#	ax.set_ylim((y[0],y[-1]))
	#	ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
	#	ax.set_ylabel(r"$y$", fontsize=fs["fsa"])
	#	ax.set_title(r"calculation")

		##-------------------------------------------------------------------------

		plt.show()
	
		exit()

	##=============================================================================

	## 1D potential for NLIN with pressure key
	if 0:

		R = 2.0
		S = 1.0
		
		x = np.linspace(-R-3,R+3,1000)
		fx = force_nlin([x,0],R,S)[0]
		U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()

		fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
		ax.plot(x, U, "-", lw=2)
		ax.axvline(-R, c="k",ls="-")
		ax.axvline(S, c="k",ls="-")
		ax.axvline(R, c="k",ls="-")
		ax.text(0.5*(x[0]-R),0.4*U.max(), r"$\leftarrow P_U$", fontsize=fs["fsl"], horizontalalignment="center")
		ax.text(0.5*(-R+S),0.4*U.max(), r"$P_T \rightarrow$", fontsize=fs["fsl"], horizontalalignment="center")
		ax.text(0.5*(S+R),0.55*U.max(), r"$\leftarrow P_S$", fontsize=fs["fsl"], horizontalalignment="center")
		ax.text(0.5*(R+x[-1]),0.55*U.max(), r"$P_R \rightarrow$", fontsize=fs["fsl"], horizontalalignment="center")

		ax.set_xlim((x[0],x[-1]))
		ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
		ax.set_ylabel(r"$U$", fontsize=fs["fsa"])
		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([])
		ax.grid()

		plt.show()
		exit()

	##=============================================================================

	## 1D potential for MLIN with regions shaded and pressure key
	if 0:

		R = 1.5
		S = 1.0
		T = 0.0
		
		x = np.linspace(-R-1,R+1,1000)
		fx = force_mlin([x,0],R,S,T)[0]
		U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()

		fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
		ax.plot(x, U, "-", lw=2)
		ax.axvline(-R, c="k",ls="-")
		ax.axvline(S, c="k",ls="-")
		ax.axvline(0.5*(S+T), c="k",ls="-")
		ax.axvline(T, c="k",ls="-")
		ax.axvline(R, c="k",ls="-")
		ax.text(0.5*(x[0]-R),	0.5*U.max(), r"$\leftarrow P_U$", fontsize=fs["fsl"], horizontalalignment="center")
		ax.text(0.5*(T+0.5*(T+S)),0.3*U.max(), r"$P_T \rightarrow$", fontsize=fs["fsl"], horizontalalignment="center")
		ax.text(0.5*(0.5*(T+S)+S),0.3*U.max(), r"$\leftarrow P_S$", fontsize=fs["fsl"], horizontalalignment="center")
		ax.text(0.5*(R+x[-1]),	0.5*U.max(), r"$P_R \rightarrow$", fontsize=fs["fsl"], horizontalalignment="center")

		ax.set_xlim((x[0],x[-1]))
		ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
		ax.set_ylabel(r"$U$", fontsize=fs["fsa"])
		ax.xaxis.set_ticklabels([])
		ax.yaxis.set_ticklabels([])
		ax.grid()

		plt.show()
		exit()
	##=============================================================================

	## 1D potential for CLIN with regions shaded and pressure key
	#def UP_CL(ax=None,R=None,S=None,T=None):
	def UP_CL(ax,R,S,T):
		"""
		Plot interior walls of CL potential with Pin Pout annotation.
		"""
	
		print ax
		if ax==None:
			print gds
			fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
		if R==None:
			R = 2.0
			S = 1.0
			T = 0.0
		
		x = np.linspace(-R,R,1000)
		fx = force_clin([x,0],R,S,T)[0]
		U = -sp.integrate.cumtrapz(fx,x,initial=0.0); U-=U.min()
	
		ax.plot(x, U, "k-", lw=2)
		
		## Pout right
		ax.text(0.6, 0.70*U.max(), r"$\mathbf{\Leftarrow}$",
			fontsize=fs["fsa"], horizontalalignment="left", color="b")
		ax.text(0.6, 0.75*U.max(), r"$\mathbf{\Leftarrow P_{\rm out}}$",
			fontsize=fs["fsa"], horizontalalignment="left", color="b")
		ax.text(0.6, 0.80*U.max(), r"$\mathbf{\Leftarrow}$",
			fontsize=fs["fsa"], horizontalalignment="left", color="b")
	
		## Pin right left
		ax.text(0, 0.70*U.max(), r"$\mathbf{\Leftarrow \qquad \Rightarrow}$",
			fontsize=fs["fsa"], horizontalalignment="center", color="r")
		ax.text(0, 0.75*U.max(), r"$\mathbf{\Leftarrow P_{\rm in}\Rightarrow}$",
			fontsize=fs["fsa"], horizontalalignment="center", color="r")
		ax.text(0, 0.80*U.max(), r"$\mathbf{\Leftarrow \qquad \Rightarrow}$",
			fontsize=fs["fsa"], horizontalalignment="center", color="r")
		
		## Pout left
		ax.text(-0.6, 0.70*U.max(), r"$\mathbf{\Rightarrow}$",
			fontsize=fs["fsa"], horizontalalignment="right", color="b")
		ax.text(-0.6, 0.75*U.max(), r"$\mathbf{P_{\rm out} \Rightarrow}$",
			fontsize=fs["fsa"], horizontalalignment="right", color="b")
		ax.text(-0.6, 0.80*U.max(), r"$\mathbf{\Rightarrow}$",
			fontsize=fs["fsa"], horizontalalignment="right", color="b")

		ax.set_xlim(x[0],x[-1])
		ax.set_ylim(0,1.2*ax.get_ylim()[1])
		ax.set_xlabel(r"$x$", fontsize=fs["fsa"])
		ax.set_ylabel(r"$U$", fontsize=fs["fsa"])
		ax.xaxis.set_major_locator(NullLocator())
		ax.yaxis.set_major_locator(NullLocator())
	
		return
	
	if 1:
		from LE_CPressure import UP_CL
		fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
		R = 2.0
		S = 1.0
		T = 0.0
		UP_CL(fig,ax,R,S,T)
		plt.show()


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
