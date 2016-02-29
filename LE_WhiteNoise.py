import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time
import os
from sys import argv

from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	Presumably this runs a white noise simulation in WN variables.
	1D -- x and eta.
	"""
	t0 = time()

	tmax = 100.0
	dt = 0.05
	Nstep = int(tmax/dt)
	Nrun = int(3e6*dt)
	w = 10.0		## Wall position
	widx = 100		## Wall index / half-length of xbins

	xmin = 0.9*w
	xmax = w+4.0
	x0 = 0.95*w		## Injection coordinate

	outfile = "Pressure/160229_WhiteNoise/WhiteNoise_r"+str(int(Nrun))+"_dt"+str(dt)
	assert os.path.isdir(os.path.dirname(outfile))

	xbins = np.concatenate([np.linspace(xmin,w,widx+1),np.linspace(w,xmax,widx+1)[1:]])
	ybins = np.linspace(-0.5,0.5,41)
	
	try:
		X,Y = np.load(argv[1])
		pdf_plot(X,Y,xbins,ybins,outfile)
	except (IndexError, IOError):
		t0 = time()
		X,Y = sim(Nrun,Nstep,dt,xmin,x0,w,outfile)
		print "Simulate",round(time()-t0,1),"seconds."
		pdf_plot(X,Y,xbins,ybins,outfile)
	
	plt.show() 
	
	return


##=============================================================================
def sim(Nrun,Nstep,dt,xmin,x0,w,outfile):

	X = []
	Y = []
	
	sd = np.sqrt(1.0/dt)

	for j in range(Nrun):

		xi = np.sqrt(2) * np.random.normal(0.0, sd, Nstep)
		x = np.zeros(Nstep)
		x[0] = x0
		xt = x[0]
		i = 0
		
		while x[i]>xmin:
			i += 1
			x[i] = x[i-1] + dt*(-0.5*(1.0+np.sign(x[i-1]-w)) +xi[i-1])
			if (i+1)%Nstep==0:
				print "Run",[j],"extending"
				x = np.append(x,np.zeros(Nstep))
				xi = np.append(xi,np.sqrt(2)*np.random.normal(0.0, sd, Nstep))
		x = x[:i]
		xi = xi[:i]
		
		X = np.append(X,x)
		Y = np.append(Y,xi)
	
	np.save(outfile,np.vstack([X,Y]))
	
	return X,Y

	## ----------------------------------------------------
	
##=============================================================================
def pdf_plot(X,Y,xbins,ybins,outfile):
	
	t1 = time()
	
	xmin = xbins[0]
	xmax = xbins[-1]
	widx = xbins.size/2
	w = xbins[widx]
	x0idx = widx/2
	
	fig, axs = plt.subplots(2,sharex=True)

	h2d,xe,ye,im = axs[0].hist2d(X,Y,[xbins,ybins],cmax=0.8, normed=True)
	h1d,xe,pa = axs[1].hist(X,xbins,normed=True,stacked=True)
	
	print "Histogram",round(time()-t1,1),"seconds."
	t1 = time()
	
	axs[0].axvline(w,color="k",linestyle="-",linewidth="3")
	axs[0].axvline(xbins[x0idx+1],color="k",linestyle="-",linewidth="3")
	axs[0].set_xlim(left=xmin,right=xmax)
	axs[0].set_ylabel("$\\eta$",fontsize=fsa)
	
	h = 1.0/(0.75+1-np.exp(w-xmax))
	axs[1].plot(xbins[:x0idx],h*np.linspace(0.0,1.0,x0idx),"r-")
	axs[1].plot(xbins[x0idx:widx],h*np.ones(widx-x0idx),"r-")
	axs[1].plot(xbins[widx:],h*np.exp(w-xbins[widx:]),"r-")
	axs[1].axvline(w,color="k",linestyle="-",linewidth="3")
	axs[1].axvline(xbins[x0idx],color="k",linestyle="-",linewidth="3")
	axs[1].set_xlim(left=xmin,right=xmax)
	axs[1].set_ylim(bottom=0.0,top=1.0)
	axs[1].set_xlabel("$x$",fontsize=fsa)
	axs[1].set_ylabel("$\\rho$",fontsize=fsa)
	axs[1].grid()
	
	print "Plot",round(time()-t1,1),"seconds."
	plt.savefig(outfile+".png")

	return

## ============================================================================	
def tail_plot(xbins, h1d, w, outfile):

	fig, ax = plt.subplots(1)
	xcent = 0.5*(xbins[1:]+xbins[:-1])
	
	# zidx = np.where(h1d==0.0)
	x_new,tail_fit,m = exp_fit(xcent[100:]-w,h1d[100:])
	
	ax.plot(xcent[100:],h1d[100:])
	ax.plot(x_new,tail_fit)
	ax.set_yscale("linear")
	ax.set_xlim(left=w,right=xbins[-1])
	ax.set_xlabel("$x$")
	ax.grid()	
	
	plt.savefig(outfile+"_T.png")
	
	return

	

## ============================================================================	
def exp_fit(x,y):
	fitfunc = lambda x,m: y[0]*np.exp(-m*x)
	popt, pcov = curve_fit(fitfunc, x, y)
	x_new = np.linspace(x[0],x[-1],5*x.size)
	return x_new, fitfunc(x_new, *popt), -popt[0]
	



## ============================================================================	
	
if __name__=="__main__":
	main()