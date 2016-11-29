import numpy as np
import os, time, subprocess
from datetime import datetime

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
	

fs = {"fsa":30,"fsl":26,"fst":20,"fsn":26,"figsize":(10,10)}
# fs = [14,12,14]
# figsize = (4,4)

def set_mplrc(fs):
	"""
	Set MPL defaults
	"""
	import matplotlib as mpl
	mpl.rcParams['xtick.labelsize'] = fs["fsn"]
	mpl.rcParams['ytick.labelsize'] = fs["fsn"]
	mpl.rc("lines", linewidth=2, markersize=8)
	mpl.rc("legend", fontsize=fs["fsl"], framealpha=0.5, fancybox=True)
	mpl.rc("savefig", dpi=200, format="pdf")
	return

	
##==========================================
## INPUT / OUTPUT
##==========================================

def boundaryfilenames(b,X,ym,nb,nr):
	""" Returns the filename for boundary pdfs, trajectories and histograms. """
	head = "./dat_LE_stream/b="+str(b)+"/"
	tail = "_y"+str(ym)+"bi"+str(nb)+"r"+str(nr)+"b"+str(b)+"X"+str(int(X))+"seed65438"
	histfile = head+"BHIS"+tail+".npy"
	trafile = head+"BTRA"+tail+".png"
	pdffile = head+"BPDF"+tail+".png"
	strfile = head+"BSTR"+tail+".png"
	return histfile, trafile, pdffile, strfile
		
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
	start = filename.find("_dt") + 3
	finish = start + 1
	while unicode(filename.replace(".",""))[start:finish].isnumeric():
		finish += 1
	dt = float(filename[start:finish])
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
		S = 0.0
	## lambda lengthscale
	try:
		start = filename.find("_l") + 2
		lam = float(filename[start:filename.find("_",start)])
	except ValueError:
		lam = -1.0
	## nu multiplier
	try:
		start = filename.find("_n") + 2
		nu = float(filename[start:filename.find("_",start)])
	except ValueError:
		nu = None
	## force type
	if "_C_" in filename:	ftype = "const"
	elif "_L_" in filename: 	ftype = "lin"
	elif "_LC_" in filename: ftype = "lico"
	elif "_DC_" in filename: ftype = "dcon"
	elif "_DL_" in filename: ftype = "dlin"
	elif "_T_" in filename:	ftype = "tan"
	elif "_N_" in filename: ftype = "nu"
	elif "_DN_" in filename: ftype = "dnu"
	else: raise ValueError, me+"Unsupported potential."
	## Geometry
	if ("_CIR_" in filename or "_POL_" in filename): geo = "POL"
	elif "_1D_" in filename: geo = "1D"
	elif "_2D_" in filename: geo = "2D"
	## Collect into lists
	pars  = [a,X,D,dt,R,S,lam,nu,ftype,geo]
	names = ["a","X","D","dt","R","S","lam","nu","ftype","geo"]
	##
	return dict(zip(names,pars))


def filename_par(filename, searchstr):
	"""
	Scrape filename for parameter
	"""
	start = filename.find(searchstr) + len(searchstr)
	finish = start + 1
	while unicode(filename[start:].replace(".",""))[:finish-start].isnumeric():
		finish += 1
	return float(filename[start:finish])


## ====================================================================

def check_path(histfile, vb):
	"""
	Check whether directory exists; and if existing file will be overwritten.
	"""
	me = "LE_Utils.check_path: "
	if os.path.isfile(histfile):
		raise IOError(me+"file",histfile,"already exists. Not overwriting.")
	try:
		assert os.path.isdir(os.path.dirname(histfile))
	except AssertionError:
		os.mkdir(os.path.dirname(histfile))
		if vb: print me+"Created directory",os.path.dirname(histfile)
	return
	
def create_readme(histfile, vb):
	"""
	If no readme exists, make one.
	NOTE commit is the LAST COMMIT -- maybe there have been changes since then.
	Assumes directory exists.
	"""
	me = "LE_Utils.create_readme: "
	readmefile = os.path.dirname(histfile)+"/README.txt"
	try:
		assert os.path.isfile(readmefile)
	except AssertionError:
		now = str(datetime.now().strftime("%Y-%m-%d %H.%M"))
		commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
		header = "Time:\t"+now+"\nCommit hash:\t"+commit+"\n\n"
		with open(readmefile,"w") as f:
			f.write(header)
		if vb: print me+"Created readme file "+readmefile
	return

## ====================================================================
