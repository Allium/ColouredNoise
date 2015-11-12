import numpy as np
import matplotlib.pyplot as plt
import os, time

"""
NAME
LE_Utils.py

PURPOSE
Supporting functions for LE_BoundarySimPlt.py

EXECUTION
None

STARTED
04 May 2015

"""

##================================================
## SETUP
##================================================

## Potentials
	
def FHO(x,b,X):
	""" Force at position x for harmonic confinement """
	return -b*x

def FBW(x,b,X):
	""" Force at position x for bulk+wall confinement """
	return -b*(np.abs(x)>=X).astype(int)*np.sign(x)
	
## Currents

def J_BW(P,b,F,Y):
	""" FPE currents: valid for BW scenario """
	dy = Y[1,0]-Y[0,0]
	Jx = (F + Y)*P
	Jy = - Y*P - b*np.gradient(P,dy)[0]
	return Jx, Jy
	

##================================================
## PLOTTING
##================================================
		
def plot_walls(ax, X, xmax,ymax, lw=2):
	"""
	Plot potential walls
	"""
	ax.plot([-X,-X],[-ymax,ymax],"r--",[X,X],[-ymax,ymax],"r--",linewidth=2)
	return

	
def Px_WBratio(Px,x,X):
	""" Roughly calculates the ratio of the "spike" to the bulk """
	Xloc = np.abs(x-X).argmin()
	avbulk = Px[-Xloc:Xloc].mean()
	avspke = np.sort(Px)[-2:].mean()
	return round(avspke/avbulk,2)

	
##==========================================
## INPUT / OUTPUT
##==========================================

def boundaryfilenames(b,X,ym,nb,nr):
	""" Returns the filename for boundary pdfs, trajectories and histograms. """
	head = "./dat_LE_stream/b="+str(b)+"/"
	tail = "_y"+str(ym)+"bi"+str(nb)+"r"+str(nr)+"b"+str(b)+"X"+str(int(X))+"seed65438"
	hisfile = head+"BHIS"+tail+".npy"
	trafile = head+"BTRA"+tail+".png"
	pdffile = head+"BPDF"+tail+".png"
	strfile = head+"BSTR"+tail+".png"
	return hisfile, trafile, pdffile, strfile
		
def save_data(outfile,data,vb=False):
	""" Write .npy file of data. File must read with np.load() """	
	me = "LE_Utils.save_data: "
	t0 = time.time()
	np.save(outfile,data)
	if vb:	print me+"Data saved to",outfile+".npy. Time",round(time.time()-t0,1),"seconds."	
	return outfile+".npy"