import numpy as np
from sys import argv
from matplotlib import pyplot as plt

from LE_Utils import fs, set_mplrc

## MPL defaults
set_mplrc(fs)

##=================================================================================================
def main(outfile=None):
	"""
	PURPOSE
	Plots PDF and streamlines for coloured-noise particle in harmonic potential
	
	EXECUTION
	python LE_ExactHO.py b
	
	EXAMPLE
	python LE_ExactHO.py 1.0
	
	STARTED
	20 March 2015
	
	DEPRECATED
	09 April 2015 -- added functionality to call from LE_Plot for subplot comparison
	"""

	me = "LE_ExactHO.main: "
	
	try: b = float(argv[1])
	except IndexError: b = 0.1
	jx0  = b
	
	xmax = 3/np.sqrt(1+b)
	ymax = 3/np.sqrt(b)
	N = 200
	x = np.linspace(-xmax, xmax, N+1)
	y = np.linspace(-ymax, ymax, N+1);
	X, Y = np.meshgrid(x, y, indexing="ij")
	
	## PDF
	rho = P_HO(X,Y,b)

	## 2D current
	Jx, Jy = J_HO(X,Y,b,rho)	
	J = np.sqrt(Jx*Jx + Jy*Jy)
	
	##---------------------------------------------------------------------------------------------
	## PLOTTING
	
	plt.rcParams["image.cmap"] = "Greys"#"coolwarm"
	
	##---------------------------------------------------------------------------------------------
	"""
	## LABEL AXES
	
	fig,axs = plt.subplots(2,2)
	fig.canvas.set_window_title("Densities")
	
	## Q(x)
	ax = axs[0][0]
	
	Qx = np.trapz(rho,y,axis=1)
	ax.plot(x, Qx, "k-")
	ax.fill_between(x, 0.0, Qx, color="k", alpha=0.2)
	
	ax.set_xlim(x[0],x[-1])
	ax.set_ylim(bottom=0.0)
	# ax.set_ylabel(r"$\eta$")
	
	## rho(x,y)
	ax = axs[1][0]
	
	ax.contourf(x, y, rho, 15)
	
	ax.set_xlim(x[0],x[-1])
	ax.set_ylim(y[0],y[-1])
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	
	## q(eta)
	ax = axs[1][1]
	
	qe = np.trapz(rho,x,axis=0)
	ax.plot(y, qe, "k-", orientation=u"horizontal")
	ax.fill_between(y, 0.0, qe, color="k", alpha=0.2)
	
	ax.set_xlim(y[0],y[-1])
	ax.set_ylim(bottom=0.0)
	# ax.set_ylabel(r"$\eta$")
		
	## Other axis
	axs[0][1].set_visible(False)
	
	plt.show()
	"""
	##---------------------------------------------------------------------------------------------
	fig = plt.figure()
	fig.canvas.set_window_title("Density")
	ax  = fig.add_subplot(111)
	
	ax.contourf(x, y, rho, 15)
	
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	
	##---------------------------------------------------------------------------------------------
		
	fig = plt.figure()
	fig.canvas.set_window_title("Streamlines")
	ax  = fig.add_subplot(111)
	
	ax.contourf(X, Y, J, 5)
	# ax.streamplot(x, y, Jx, Jy, linewidth=4.0*np.sqrt(J/J.max()))
	stp=10
	Jx /= rho
	Jy /= rho
	ax.quiver(x[::stp], y[::stp], Jx[::stp,::stp], Jy[::stp,::stp])
	
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	
	
	plt.show()
	
	return


##=================================================================================================

def P_HO(x,y,b):
	return b*np.sqrt(b+1)/(2*np.pi)*np.exp(-0.5*(b+1)**2*x*x -0.5*b*(b+1)*y*y + b*(b+1)*x*y)
	
def J_HO(X,Y,b,P):	
	Jx = -(Y-X)*P
	Jy = -1/b*Y*P - 1/(b*b)*np.gradient(P)[1]
	return (Jx, Jy)
	
##=================================================================================================
if __name__=="__main__":
	main()