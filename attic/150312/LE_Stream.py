import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import csv
import time
import os
from sys import argv
from scipy.interpolate import griddata
from scipy.ndimage.filters import gaussian_filter
from math import floor

import LE_Simulate



def main():
	"""
	STARTED
	02 March 2015

	DESCRIPTION
	Reads in LE data (csv) and makes streamline plots.
	
	EXAMPLE
	python LE_Stream.py dat_LE_stream/HO_b1X10s1n200000seed65438.csv

	BUGS / TO DO
	- histogram: Assign bin width cleverly
	- pdf: tidy up plots
	"""
	
	datafile = None
	try: datafile = argv[1]
	except IndexError:
		raise IOError("LE_Stream.main: Need an input file!")
		print main.__doc__
	
	LEstr(datafile)

	return

##================================================

def LEpdf(datafile=None):
	""" Plot pdf """
	
	## Read eta (yy), xHO (x1) points from file
	yy,x1 = np.loadtxt(datafile,delimiter=" ",skiprows=1)[:,1:3].T

	## Construct a (normed) histogram of the data
	nbins = [100,100]
	H,xedges,yedges = np.histogram2d(x1,yy,bins=nbins,normed=True)
	xpos = xedges[1:]-xedges[0]; ypos = yedges[1:]-yedges[0]

	## Plot pdf
	H = gaussian_filter(H, 3)	## Convolve with Gaussian
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3)
	ax1.imshow(H, interpolation='nearest', origin='low')#,extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
	ax1.set_xlabel("$x_{HO}$");ax1.set_ylabel("$\eta$")
	ax2.contour(xpos,ypos,H,10)
	ax2.set_xlabel("$x_{HO}$");ax2.set_ylabel("$\eta$")
	ax3.hist2d(x1,yy, bins=100, normed=True)
	ax3.set_xlabel("$x$");ax3.set_ylabel("$\eta$")
	plt.tight_layout()

	plt.show()
	return

##================================================

def LEstr(datafile=None):
	""" Plot streamlines """
	
	nm = "LE_Stream.LEstr: "
	
	if datafile==None:
		try: datafile = argv[1]
		except IndexError: raise IOError(nm+"Need an input file!")
	
	## Analysis options
	for i in range(0,len(datafile)):
		if datafile[i]=="b": b = float(datafile[i+1:i+4]);	break
	for i in range(0,len(datafile)):
		if datafile[i]=="X":	j=i
		if datafile[i]=="s" and datafile[i+1]=="1":	X = float(datafile[j+1:i]);	break
	
	## Output options
	saveplot = True
	showplot = False
	outfile = os.path.split(datafile)[0]+"/stream_"+os.path.split(datafile)[1][0:-4]
	
	t0 = time.time()
	## Read eta (yy), xHO (x1) points from file
	yy, x1 = np.loadtxt(datafile,delimiter=" ",skiprows=1)[:,1:3].T
	print nm+"Reading data",round(time.time()-t0,1),"seconds"
	coords = zip(x1,yy)
	
	## Set up grid of points in x-y
	gridsize = 20	
	#x = np.linspace(-2*X,2*X, gridsize);	y = np.linspace(-2*b,2*b, gridsize)
	x = np.linspace(-15,15, gridsize);	y = np.linspace(-1.5,1.5, gridsize)
	xi,yi = np.meshgrid(x,y)
	#print xi;return
	
	## Calculate speeds (1D arrays)
	vx1 = np.diff(x1);vx1=np.append(np.array(vx1[0]),vx1)
	vyy = np.diff(yy);vyy=np.append(np.array(vyy[0]),vyy)
	vyy/=100	## HACK TO RE-SCALE ETA -- MESSY!
	v1  = np.sqrt(vx1**2+vyy**2)	
	del x1, yy
		
	t0 = time.time()
	## Interpolate data onto grid
	gvx11 = griddata(coords, vx1, (xi,yi), method='nearest')
	gvyy1 = griddata(coords, vyy, (xi,yi), method='nearest')
	gv1   = griddata(coords, v1,  (xi,yi), method='nearest')
	print nm+"Gridding data",round(time.time()-t0,1),"seconds"
	del coords
	
	if saveplot or showplot:
		## Subplots for 1 and 2
		fig,ax1 = plt.subplots(1,1)
		fig.suptitle(outfile)
		fig.set_facecolor("white")

		## Smooth data
		smooth= 2
		gvyy1 = gaussian_filter(gvyy1, smooth)
		gvx11 = gaussian_filter(gvx11, smooth)
		gv1   = gaussian_filter(gv1, smooth)
		## Line widths
		lw1 = 3.0*gv1/gv1.max()
		
		t0 = time.time()
		## Make plots
		fs = 25
		ax1.contourf(xi,yi,gv1, 4, alpha=0.4)
		ax1.streamplot(x,y, gvx11,gvyy1, linewidth=lw1, cmap=plt.cm.jet)
		#ax1.plot([-X,-X],[-2*b,2*b],"k--",linewidth=2);ax1.plot([X,X],[-2*b,2*b],"k--",linewidth=2)
		ax1.plot([-X,-X],[-1.5,1.5],"k--",linewidth=2);ax1.plot([X,X],[-1.5,1.5],"k--",linewidth=2)
		ax1.set_xlabel("$x$",fontsize=fs);ax1.set_ylabel("$\eta$",fontsize=fs)
		print nm+"Plotting",round(time.time()-t0,1),"seconds"
		## NEED: colorbar
		
		if saveplot:
			fig.savefig(outfile+"TEST.png",facecolor=fig.get_facecolor(), edgecolor="none")
			print nm+"Plot saved",outfile+".png"
		if showplot:
			plt.show()

	return

##================================================

def LE_SimStr(b,X):
	"""
	A module to produce data from simulation and analyse it to streamplot.
	"""
	
	datafile = LE_Simulate.LEsim(b,X)
	LEstr(datafile)
	
	return


##================================================

def LE_SimStr_auto():
	
	for b in [0.1,0.5,1.0]:
		for X in [1,5,10]:
			LE_SimStr(b,X)
	return
	
##================================================	

def histogram_data(datafile):
	"""
	Construct a histogram from a large CSV file by iterating.
	"""

	with open(datafile, "rb") as csvfile:
		reader = csv.reader(csvfile, delimiter=' ')

		npnts = sum(1 for row in reader)
		nbins = npnts/10
		xmax=5.0;xmin=-xmax;ymax=xmax;ymin=-ymax
		binwidth = 2*xmax / nbins
		xbins = np.linspace(xmin,xmax,nbins) 
		ybins = np.linspace(ymin,ymax,nbins)

		histarr = np.zeros([nbins,nbins])
		for row in reader:
			bin = (int(floor(row[0]/binwidth)),int(floor(row[1]/binwidth)))
			histarr[bin] += 1
		
	return (histarr,xbins,ybins)

##================================================
def denan(arrs):
	for arr in arrs: arr[np.isnan(arr)]=0.0
	return

##================================================
##================================================
if __name__=="__main__":
	main()
