import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from sys import argv


def main():
	"""
	STARTED
	07/03/2015
	
	PURPOSE
	Simulate a Langevin process.
	Output data to file.
	
	INPUTS
	b: force level
	X: wall postion
	
	EXECUTION
	python LE_Simulate.py b X BWpot
	
	EXAMPLE
	python LE_Simulate.py 1.0 5.0 1

	BUGS / TODO
	- trapz integration
	"""
	
	## Read parameter values from CLAs
	try: b = float(argv[1])
	except IndexError: b=1.0	
	try: X = float(argv[2])
	except IndexError: X=5.0
	try: BWpot = int(argv[3])
	except IndexError: BWpot=1
	
	## Run simulation; returns datafile path
	out = LEsim(b,X,BWpot)
	
	return out

##================================================

def LEsim(b=None, X=None, BWpot=1):
	"""
	Run the LE simulation.
	"""

	t0 = time.time()
	nm = "LE_Simulate.LEsim: "

	## System parameters
	s = 1.0	## Noise level
	if b is None:
		try: b = float(argv[1])
		except IndexError: b=1.0
	if X is None:
		try: X = float(argv[2])
		except IndexError: X=5.0
	## Choose potential
	if BWpot:	force = FBW; pot = "BW"
	else:			force = FHO; pot = "HO"
	
	## Simulation parameters
	dt = 0.05
	tmax = 1.0*10**4
	nstp = int(tmax/dt)
	tarr = np.arange(0,tmax,dt)

	## Initialisation
	eta0 = 0.0
	x0 = 0.0
	seed = 65438; np.random.seed(seed)
	xi = np.random.normal(0,s,nstp)

	## Output options
	showplot = False
	saveplot = True
	savedata = True
	outfile  = "./dat_LE_stream/"+pot+"_b"+str(b)+"X"+str(int(X))+"s"+str(int(s))+"n"+str(nstp)+"seed"+str(seed)
	
	## OU noise
	t0=time.time()
	expmt = np.exp(-tarr)	## Precompute exp(-t) array
	eta = eta0*expmt
	eta[0] += xi[0]
	lbound = int(35/dt)	## Set exp(-35) to zero -- only 700 terms
	for i in range(1,nstp):
		j = min(i,lbound)
		eta[i] += expmt[:j][::-1].dot(xi[i-j:i])*dt
	print nm+"Simulation of eta",round(time.time()-t0,1),"seconds for",nstp,"steps";t0 = time.time()

	## Variable of interest
	x = dt*np.cumsum( eta )
	x[0] += x0; fdtsum=0
	for i in range(1,nstp):
		fdtsum += dt*force(x[i-1],b,X)
		x[i] += fdtsum
	print nm+"Simulation of x  ",round(time.time()-t0,1),"seconds for",nstp,"steps";t0 = time.time()

	## Plot walk
	if showplot or saveplot:
		fs = 25
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
		fig.suptitle(outfile+"\n .")
		envelope_plot(tarr, xi,  winsize=tmax/80, ax=ax1)
		ax1.set_xlabel("$t$",fontsize=fs);ax1.set_ylabel("$\\xi$",fontsize=fs)
		envelope_plot(tarr, eta, winsize=tmax/80, ax=ax2)
		ax2.set_xlabel("$t$",fontsize=fs);ax2.set_ylabel("$\eta$",fontsize=fs)
		envelope_plot(tarr, x, winsize=tmax/80, ax=ax3)
		#ax3.plot(tarr, x)
		ax3.set_xlabel("$t$",fontsize=fs);ax3.set_ylabel("$x$",fontsize=fs)
		etalim = np.ceil(abs(eta).max())	## Not perfect
		ax4.hist2d(x,eta, bins=100, range=[[-2*X,+2*X],[-2*b,+2*b]], normed=True)
		ax4.set_xlabel("$x$",fontsize=fs);ax4.set_ylabel("$\eta$",fontsize=fs)
		fig.tight_layout()
		if saveplot:
			plt.savefig(outfile+".png")
			print nm+"Plot saved",outfile+".png"
		print nm+"Making plot",round(time.time()-t0,1),"seconds"
		if showplot:
			plt.show()	
		plt.close(fig)

	## Write to data file
	if savedata:
		t0 = time.time()
		with open(outfile+".csv", "wb") as csvfile:
			writer = csv.writer(csvfile, delimiter=' ', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(["time","eta","x"])
			for row in zip(tarr,eta,x):	writer.writerow(row)
		print nm+"Data saved",outfile+".csv"
		print nm+"Saving data",round(time.time()-t0,1),"seconds for",nstp,"points";t0 = time.time()
	
	return outfile+".csv"

##================================================
	
def FHO(x,b,X):
	""" Force at position x for harmonic confinement """
	return -b*x

def FBW(x,b,X):
	""" Force at position x for bulk+wall confinement """
	f = -b*np.sign(x) if abs(x)>X else 0.0
	#if x is List: [f = FBW[xi] for xi in x]
	return f

##================================================

def automate():
	""" Loop over parameters """	

	for b in [0.0,1.0,5.0,10.0,20.0]:
		for X in [1.0,2.0,5.0,10.0]:
			LEsim(b, X)

	return

##================================================

def envelope_plot(x, y, winsize, ax=None, fill='gray', color='blue'):
	""" Plot smoothed mean with grey envelope """
	if ax is None:	ax = plt.gca()
	# Coarsely chunk the data, discarding the last window if it's not evenly
	# divisible. (Fast and memory-efficient)
	numwin = x.size // winsize
	ywin = y[:winsize * numwin].reshape(-1, winsize)
	xwin = x[:winsize * numwin].reshape(-1, winsize)
	# Find the min, max, and mean within each window 
	ymin = ywin.min(axis=1)
	ymax = ywin.max(axis=1)
	ymean = ywin.mean(axis=1)
	xmean = xwin.mean(axis=1)

	fill_artist = ax.fill_between(xmean, ymin, ymax, color=fill, 
		                  edgecolor='none', alpha=0.5)
	line, = ax.plot(xmean, ymean, color=color, linestyle='-')
	return fill_artist, line

##================================================
##================================================
if __name__=="__main__":
	main()
