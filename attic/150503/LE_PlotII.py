import numpy as np
import matplotlib.pyplot as plt
import time,os
import optparse
from scipy.ndimage.filters import gaussian_filter
from sys import argv

import LE_Simulate
from LE_Plot import plot_pdf, get_filenames, plot_separatrix, plot_walls
# from LE_ExactHO import main as plot_exact
 
 
def main():
	"""
	PURPOSE
	Calculate and plot currents in x-eta space for a given realisation of a random walk.
	
	BACKGROUND
	In LE_Plot.py, currents were painstakingly constructed by interpolating random data
	onto a fixed grid and making a streamplot. Here I simply calculate currents at each
	point in space from the numerically-determined PDF.
	
	STARTED
	13/04/2015
	
	EXECUTION
	
	OPTIONS
	
	EXAMPLE
	
	BUGS
	"""
	me = "LE_PlotII.main: "
	t0 = time.time()
	
	## User-input parameters
	
	try: b = float(argv[1])
	except IndexError: b = 1.0
	
	try: X = float(argv[2])
	except IndexError: X = 1.0
		
	parser = optparse.OptionParser()
	
	parser.add_option('-t','--timefac',
                  dest="timefac",
                  default=1.0,
                  type="float",
                  )
	parser.add_option('--BW',
                  dest="BW",
                  default=False,
                  action="store_true"
                  )
				  
	parser.add_option('-s','--smooth',
                  dest="smooth",
                  default=1.0,
                  type="float",
                  )
				  
	opt = parser.parse_args()[0]
	
	timefac = opt.timefac
	BW		= opt.BW
	smooth	= opt.smooth
	
	print "\n== "+me+"b =",b," X =",X," BW =",BW,"==\n"
	if not BW: print me+"Warning: calling without --BW option is meaningless. Continuing..."
	
	plot_stream_fromPDF(b,X,timefac,BW,smooth)
	
	print me+"Total time",round(time.time()-t0,1),"seconds"
	
	return
	
##=======================================================================

def plot_stream_fromPDF(b,X,timefac,BW,smooth):
	"""
	Create PDF from trajectory, calculate current field, and plot.
	"""
	me = "LE_PlotII.plot_stream_fromPDF: "
	
	showplot = False
	
	## Filenames
	trafile, rndfile, pdffile, strfile, n = get_filenames(b,X,timefac,BW)
	
	## Read in trajectory data (or simulate anew)
	try:		
		xydata = np.load(trafile+".npy")[1:3,:n]
		print me+"Trajectory file found",trafile+".npy"
	except IOError:
		print me+"No trajectory file found. Simulating..."
		xydata = LE_Simulate.LEsim(b,X,timefac,BW)[1:3,:n]
		
	## Create histogram and save PDF plot
	P, x, y = plot_pdf(xydata, b,X, pdffile)
	xmax, ymax = np.abs(x).max(), np.abs(y).max()
	x = (x[1:]+x[:-1])*0.5; y = (y[1:]+y[:-1])*0.5
	gx, gy = np.meshgrid(x, y); gy = -gy
	
	## Force field
	f = LE_Simulate.FBW(gx,b,X) if BW else LE_Simulate.FHO(gx,b,X)

	## Compute currents (on-grid)
	Jx, Jy = J_BW(P,b,f,gy)
	Jt = np.sqrt(Jx*Jx+Jy*Jy)
		
	## --------------------------------------------------------------------	
	
	## Smooth data
	if smooth is not 0.0:
		Jx = gaussian_filter(Jx, smooth)
		Jy = gaussian_filter(Jy, smooth)
		Jt = gaussian_filter(Jt, smooth)
	strfile += "_sm"+str(smooth)
		
	## --------------------------------------------------------------------	
	
	## Plotting
	
	t0 = time.time()
	fs = 25
	
	fig = plt.figure(facecolor="white")
	fig.suptitle(strfile)
	
	##############################
	P /= np.trapz(P.sum(axis=1),y)
	## Plot 1D PDFs and currents
	if True:
		plotfunc = plt.semilogy if 1 else plt.plot
		dx = x[1]-x[0]
		v = 0.051**2 if b==0.1 else 0.051**2
		# v = (P*dx).sum(axis=1).var();	print v
		Pex = 1/np.sqrt(2*np.pi*v)*np.exp(-y*y*0.5/v)
		## Plot P(y) against y
		plotfunc(y,P.sum(axis=1),"r-")
		plotfunc(y,Pex,"b--")
		plt.xlabel("$y$");plt.ylabel("$P(y)$")
		plt.savefig("Py_b"+str(b)+".png")
		plt.show() if showplot else plt.close()
		## Plot Jy against y
		plt.plot(y,Jy.sum(axis=1),"r-")
		plt.plot(y,-Pex*y*(b/v-1)/dx,"b--")
		plt.xlabel("$y$");plt.ylabel("$J_y(y)$")
		plt.savefig("Jy_b"+str(b)+".png")
		plt.show() if showplot else plt.close()
		## Plot P(x) against x
		plt.plot(x,P.sum(axis=0),"r-")
		plt.annotate("Wall-to-bulk ratio: $\sim"+str(Px_WBratio(P.sum(axis=0),x,X))+"$",
			xy=(0,0), xycoords='axes fraction', xytext=(0.8, 0.95), textcoords='axes fraction',
            horizontalalignment='right', verticalalignment='top'
            )
		plt.xlabel("$x$");plt.ylabel("$P(x)$")		
		plt.savefig("Px_b"+str(b)+".png")
		plt.show() if showplot else plt.close()
		## Plot Jx against y
		plt.plot(y,Jx.sum(axis=1),"r-")
		plt.plot(y,-Pex*y/(x[1]-x[0]),"b--")
		plt.xlabel("$y$");plt.ylabel("$J_x(y)$")
		plt.savefig("Jx_b"+str(b)+".png")
		plt.show() if showplot else plt.close()
		exit()
	##############################

	
	## Add subplot with exact solution
	if not BW:
		ax1 = fig.add_subplot(121,aspect="auto"); ax2 = fig.add_subplot(122,aspect="auto",sharey=ax1)
		plot_exact((ax2,xmax,ymax,b,False))
		# fig.tight_layout();fig.subplots_adjust(top=0.93)
		print me+"Plotting exact",round(time.time()-t0,1),"seconds"
		plot_separatrix(ax1, b, xmax, ymax, 2)
	else:
		ax1 = fig.add_subplot(111)
		plot_walls(ax1, X, xmax,ymax, 2)
	
	## Accoutrements	
	ax1.set_xlim([-xmax,xmax]);	ax1.set_ylim([-ymax,ymax])
	ax1.set_xlabel("$x$",fontsize=fs);ax1.set_ylabel("$\eta$",fontsize=fs)
	
	## Streamplot
	t0=time.time()
	lw = 2#3.0*Jt/Jt.max()
	q=1;ax1.quiver(gx[::q, ::q],gy[::q, ::q],Jx[::q, ::q],Jy[::q, ::q])
	# ax1.streamplot(gx,gy, Jx,Jy, arrowsize=1.8, arrowstyle="->", linewidth=lw, minlength=xmax/20)	
	print me+"Plotting streamlines ",round(time.time()-t0,1),"seconds"; t0=time.time()
		
	## Output
	fig.savefig(strfile+"FP.png", edgecolor="none")
	print me+"Plot saved as\n  ",strfile+"FP.png"
	plt.show()
	plt.close()
	
	return
	
##==========================================================================

def J_BW(P,b,F,Y):
	""" FPE currents: valid for BW scenario """
	dy = Y[1,0]-Y[0,0]
	Jx = (F + Y)*P
	Jy = - Y*P - b*np.gradient(P,dy)[0]
	return Jx, Jy

	
##==========================================================================

def Px_WBratio(Px,x,X):
	""" Roughly calculates the ratio of the "spike" to the bulk """
	Xloc = np.abs(x-X).argmin()
	avbulk = Px[-Xloc:Xloc].mean()
	avspke = np.sort(Px)[-2:].mean()
	return round(avspke/avbulk,2)

##==========================================================================
##==========================================================================
if __name__=="__main__":
	main()