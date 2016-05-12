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
	
def FHO(x,X):
	""" Force at position x for harmonic confinement """
	return -x

def FBW(x,X):
	""" Force at position x for bulk+wall confinement """
	# return -b*(np.abs(x)>=X).astype(int)*np.sign(x)
	## Only good for single wall
	return -0.5*(np.sign(x-X)+1)
	
def force_1D_const(x,X,D):
	"""
	Force for bulk + linear wall, but with smooth onset parameterised by Delta.
	Only good for a single wall
	"""
	if D==0:
		return FBW(x,X)
	else:
		return -0.5*(np.tanh((x-X)/(D*X))+1.0)

def force_1D_lin(x,X,D):
	"""
	Force for bulk + harmonic potential
	"""
	return 0.5*(np.sign(X-x)-1) * (x-X)
	
def force_1D_lin_jump(x,X,sh):
	return force_1D_lin(x,X,0.0) + sh
	
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
	

def plot_fontsizes():
	"""
	Axes, legend, title
	"""
	#return 18,14,18
	return 26,16,20

	
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


def filename_pars(filename):
	"""
	Scrape filename for parameters and return a dict.
	"""
	## a
	try:
		start = filename.find("_a") + 2
		a = float(filename[start:filename.find("_",start)])
	except ValueError:
		a = None
	## X
	try:
		start = filename.find("_X") + 2
		X = float(filename[start:filename.find("_",start)])
	except ValueError:
		X = None
	## D
	try:
		start = filename.find("_D") + 2
		D = float(filename[start:filename.find("_",start)])
	except ValueError:
		D = 0.0
	## dt
	# try:
		# start = filename.find("_dt") + 3
		# dt = float(filename[start:filename.find(".npy",start)])
	# except ValueError:
		# start = filename.find("_dt",start) + 3
		# dt = float(filename[start:filename.find(".npy",start)])
	# except ValueError:
	dt = 0.01
	## R
	try:
		start = filename.find("_R") + 2
		R = float(filename[start:filename.find("_",start)])
	except ValueError:
		R = None
	try:
		start = filename.find("_S") + 2
		S = float(filename[start:filename.find("_",start)])
	except ValueError:
		S = None
	## force type
	if filename.find("_C_") > 0:	ftype = "const"
	elif filename.find("_L_") > 0: 	ftype = "lin"
	elif filename.find("_LC_") > 0: ftype = "lico"
	elif filename.find("_DC_") > 0: ftype = "dcon"
	elif filename.find("_DL_") > 0: ftype = "dlin"
	elif filename.find("_T_") > 0:	ftype = "tan"
	elif filename.find("_DT_") > 0: ftype = "dtan"
	## Geometry
	if filename.find("_CIR_") > 0: geo = "CIR"
	elif filename.find("_1D_") > 0: geo = "1D"
	elif filename.find("_2D_") > 0: geo = "2D"
	## Collect into lists
	names = ["a","X","D","dt","R","S","ftype","geo"]
	pars  = [a,X,D,dt,R,S,ftype,geo]
	##
	return dict(zip(names,pars))


