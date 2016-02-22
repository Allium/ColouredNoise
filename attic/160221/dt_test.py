import numpy as np
from matplotlib import pyplot as plt
import os, glob
from LE_LightBoundarySim import *
from LE_Pressure import plot_acco

dir = "Pressure/151203X10.0r20/"


## File discovery
histfiles = np.sort(glob.glob(dir+"/*.npy"))
print histfiles
numfiles = len(histfiles)

## Assume all files have same X
start = histfiles[0].find("_X") + 2
X = float(histfiles[0][start:histfiles[0].find("_",start)])

## Outfile name
pressplot = dir+"/PressureDt.png"
Alpha = np.zeros(numfiles)
Press = np.zeros(numfiles)
	
## Loop over files
for i,filepath in enumerate(histfiles):
	
	## Find alpha
	start = filepath.find("_dt") + 3
	Alpha[i] = float(filepath[start:filepath.find(".npy",start)])
			
	## Load data
	H = np.load(filepath)
	# H /= Alpha[i]
	
	## Space
	xmin,xmax = 0.9*X,lookup_xmax(X,Alpha[i])
	ymax = 0.5
	x = calculate_xbin(xmin,X,xmax,H.shape[1]-1)
	y = np.linspace(-ymax,ymax,H.shape[0])
	
	## Marginalise to PDF in X
	Hx = np.trapz(H,x=y,axis=0)

	## Calculate pressure
	force = -Alpha[i]*0.5*(np.sign(x-X)+1)
	Press[i] = np.trapz(-force*Hx, x)
	
plt.plot(Alpha,Press,"bo-",label=".")
plt.ylim(bottom=0.0)
plot_acco(plt.gca(), xlabel="$dt$", ylabel="Pressure", legloc="")

plt.savefig(pressplot)
plt.show()