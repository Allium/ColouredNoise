import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from LE_Plot import plot_separatrix


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
	fs = 25
	
	try: b=float(argv[1])
	except IndexError: b = 0.1
	jx0  = b
	
	xmax = 2.0; ymax = 1.0
	N = 100
	x = np.linspace(-xmax, xmax, N);	y = np.linspace(-ymax, ymax, N);
	X, Y = np.meshgrid(x, y); Y = Y[::-1,:]	## Need to flip Y
	
	## Figure destination
	outdir = "dat_LE_stream/HO/"
	if outfile is None:	outfile = "HO_b"+str(b)+"_X"+str(int(xmax/2))+"_e"
	
	## PDF
	P = P_HO(X,Y,b)
	P /= 2*xmax/N*P.sum()
	
	plt.xlabel("$x$",fontsize=fs);plt.ylabel("$\eta$",fontsize=fs)
	if 0:
		plt.xlim([-xmax,xmax]);	plt.ylim([-ymax,ymax])
		plt.imshow(P, aspect=xmax/ymax, extent=[-xmax,xmax,-ymax,ymax])
		plot_separatrix(b,xmax,ymax, 2)
		plt.savefig(outdir+"PDF_"+outfile+".png")
		plt.clf()
	
	## 2D current
	Jx, Jy = J_HO(X,Y,b,P)	
	speed = np.sqrt(Jx*Jx + Jy*Jy)
	
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	
	plt.xlabel("$x$",fontsize=fs);plt.ylabel("$\eta$",fontsize=fs)
	plt.xlim([-xmax,xmax]);	plt.ylim([-ymax,ymax])
	lw = 4.0*np.sqrt(speed/speed.max())
			
	plt.contourf(X, Y, speed, 4, alpha=0.4)
	plt.streamplot(X, Y, Jx, Jy, linewidth=lw, cmap=plt.cm.jet)
	plot_separatrix(b,xmax,ymax, 3)
	
	plt.savefig(outdir+"STR_"+outfile+".png")
	
	print me+"Figures saved to",outdir
	
	return


def P_HO(x,y,b):
	return np.exp(-0.5*b*(b+1)**2*x*x -0.5*(b+1)*y*y + b*(b+1)*x*y)
	
def J_HO(X,Y,b,P):	
	Jx = -(b*X - Y)*P
	Jy = -b*(Y - (b+1)*X)*P
	return (Jx,Jy)
	
	

if __name__=="__main__":
	main()