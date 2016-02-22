import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from LE_Plot import plot_separatrix


def main(params=None,outfile=None):
	"""
	PURPOSE
	Plots PDF and streamlines for coloured-noise particle in harmonic potential
	
	EXECUTION
	python LE_ExactHO.py b
	
	EXAMPLE
	python LE_ExactHO.py 1.0
	
	STARTED
	20 March 2015
	"""

	me = "LE_ExactHO.main: "
	fs = 25
	
	PDF = False
	
	try: ax,xmax,ymax,b,outfile = params	
	except:	
		try: b=float(argv[1])
		except IndexError: b = 0.1
		xmax = 2.0; ymax = 1.0
		fig = plt.figure()
		ax  = fig.add_subplot(111)
	
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
	if PDF:
		plt.xlim([-xmax,xmax]);	plt.ylim([-ymax,ymax])
		plt.imshow(P, aspect=xmax/ymax, extent=[-xmax,xmax,-ymax,ymax])
		plot_separatrix(b,xmax,ymax, 2)
		plt.savefig(outdir+"PDF_"+outfile+".png")
		plt.clf()
	
	## 2D current
	Jx, Jy = J_HO(P,b,X,Y)	
	speed = np.sqrt(Jx*Jx + Jy*Jy)
	
	## Set up plot
	ax.set_xlabel("$x$",fontsize=fs);	ax.set_ylabel("$\eta$",fontsize=fs)
	ax.set_xlim([-xmax,xmax]);			ax.set_ylim([-ymax,ymax])
	lw = 3.0*np.sqrt(speed/speed.max())
	
	## Plotting
	ax.contourf(X, Y, speed, 4, alpha=0.4)
	ax.streamplot(X, Y, Jx, Jy, linewidth=lw)
	plot_separatrix(ax, b,xmax, ymax, 2)
	
	## Save figure
	if outfile is not False: 
		plt.savefig(outdir+"STR_"+outfile+".png")
		print me+"Figures saved to",outdir
	
	return


def P_HO(x,y,b):
	return np.exp(-0.5*b*(b+1)**2*x*x -0.5*(b+1)*y*y + b*(b+1)*x*y)
	
def J_HO(P,b,X,Y):	
	Jx = -(b*X - Y)*P
	Jy = -b*(Y - (b+1)*X)*P
	return Jx, Jy
	
	

if __name__=="__main__":
	main()