import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time
import os

def main():
	"""
	Presumably this runs a white noise simulation in WN variables.
	1D -- x and eta.
	"""
	t0 = time()

	tmax = 100.0
	dt = 0.05
	Nstep = int(tmax/dt)
	Nrun = int(5e6*dt)
	w = 10.0		## Wall position
	widx = 100		## Wall index / half-length of xbins

	xmin = 0.9*w
	xmax = w+1.0
	x0 = 0.95*w		## Injection coordinate

	outfile = "Pressure/WhiteNoise/WhiteNoise_r"+str(int(Nrun))+"_dt"+str(dt)
	assert os.path.isdir(os.path.dirname(outfile))

	xbins = np.concatenate([np.linspace(xmin,w,widx+1),np.linspace(w,xmax,widx+1)[1:]])
	ybins = np.linspace(-0.5,0.5,41)

	## ----------------------------------------------------
	
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

	print "Simulate",round(time()-t0,1),"seconds."
	t1 = time()

	## ----------------------------------------------------
	
	fig, axs = plt.subplots(2,sharex=True)

	h2d,xe,ye,im = axs[0].hist2d(X,Y,[xbins,ybins],normed=True)
	h1d,xe,pa = axs[1].hist(X,xbins,normed=True,stacked=True)
	
	print "Histogram",round(time()-t1,1),"seconds."
	t1 = time()
	
	axs[0].set_xlim(left=xmin,right=xmax)
	axs[0].set_ylabel("$\\eta$")
	axs[0].axvline(w,color="k",linestyle="-",linewidth="2")

	axs[1].plot(xbins[:widx],0.5*np.ones(widx),"r-")
	axs[1].plot(xbins[widx:],0.5*np.exp(w-xbins[widx:]),"r-")
	axs[1].axvline(w,color="k",linestyle="-",linewidth="2")
	axs[1].set_xlim(left=xmin,right=xmax)
	axs[1].set_ylim(bottom=0.0,top=1.5)
	axs[1].set_xlabel("$x$")
	axs[1].set_ylabel("$\\rho$")
	axs[1].grid()

	np.save(outfile,h2d)
	plt.savefig(outfile+".png")

	print "Plot 1",round(time()-t1,1),"seconds."
	t1 = time()
	
	## ----------------------------------------------------
	
	# tail_plot(xbins, h1d, w, outfile)

	print "Plot 2",round(time()-t1,1),"seconds."
	t1 = time()

	## ----------------------------------------------------

	plt.show() 

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