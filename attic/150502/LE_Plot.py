import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import time
import os
import optparse
from gc import collect
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
	Runs LE simulation and constructs streamplot.
	
	EXECUTION
	python LE_Plot.py b X [opts]
	
	OPTIONS
	-t --timefac	1.0		Time factor, float
	--BW			False	Selects bulk+wall setup
	
	EXAMPLE
	python LE_Plot.py 1.0 5.0 -t 2.0 --BW
	python LE_Plot.py 1.0 1.0 -t 2.0

	BUGS / TO DO
	- Need a clever way of searching for relevant data file
	- Have to speed up griddata	
	"""
	"""
	main() parses arguments to pass to SimPlt
	"""
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
 		
	LE_SimPlt(b, X, opt.timefac, opt.BW, opt.smooth)

	return

##================================================

def LE_SimPlt(b,X,timefac,BW,smooth):
	"""
	Wrapper for simulation + streamplot.
	"""
	me = "LE_Plot.LE_SimPlt: "
	
	print "\n== "+me+"b =",b," X =",X," BW =",BW,"==\n"
	
	t0 = time.time()
	
	## Filenames
	trafile, rndfile, pdffile, strfile, n = get_filenames(b,X,timefac,BW)
	
	## GET_TRADATA
	## Time-series / trajectory data: load or simulate
	try:		
		txyxidata = np.load(trafile+".npy")[:n,:]
		print me+"Trajectory file found",trafile+".npy"
	except IOError:
		print me+"No trajectory file found. Simulating..."
		txyxidata = LE_Simulate.LEsim(b,X,timefac,BW)
		
	print me,collect()
	
	## GET_STRDATA
	## Interpolated current data: load or calculate
	try:
		A = np.load(strfile+".npy"); grd = A.shape[1]
		x,y,gvx,gvy = A[0],A[1],A[2:grd+2],A[grd+2:]
		oargs = np.loadtxt(strfile+".hdr")
		print me+"Steamgrid file found",strfile+".npy"
	except IOError:
		print me+"No streamgrid file found. Calculating..."
		x,y,gvx,gvy,oargs = calc_grid(txyxidata[1:3], b,X, strfile, BW)

	print me,collect()
	
	
	## Plots
	# plot_rand(txyxidata, b,X, rndfile)
	# print me,collect()
	plot_pdf(txyxidata[1:3], b,X, pdffile)
	print me,collect()
	plot_stream( x,y,gvx,gvy,np.append(oargs,smooth), strfile )
	print me,collect()
	
	print me+"Total time",round(time.time()-t0,1),"seconds"	
	return
	
##================================================
def get_filenames(b,X,timefac,BW):
	"""
	Construct relevant filenames.
	trafile (trajectory file); outfile (destination); strfile (interpolated gridded flow data)
	"""
	n = int(200000*X*timefac); nn = int(200000*X*10)
	if BW:
		## HACK for DB space
		trafile = "C:/AsusWebStorage/MySyncData/TRAFILES/TRA_BW_b"+str(b)+"X"+str(int(X))+"n"+str(nn)+"seed65438"
		if not os.path.isfile(trafile): trafile = "./dat_LE_stream/TRA_BW_b"+str(b)+"X"+str(int(X))+"n"+str(n)+"seed65438"
		outfile = "./dat_LE_stream/b="+str(b)+"/"+os.path.split(trafile)[1][4:]
		# trafile = "./dat_LE_stream/TRA_BW_b"+str(b)+"X"+str(int(X))+"n"+str(nn)+"seed65438"
		# if not os.path.isfile(trafile): trafile = "./dat_LE_stream/TRA_BW_b"+str(b)+"X"+str(int(X))+"n"+str(n)+"seed65438"
		# outfile = os.path.split(trafile)[0]+"/b="+str(b)+"/"+os.path.split(trafile)[1][4:]
	else:
		## HACK for DB space
		trafile = "C:/AsusWebStorage/MySyncData/TRAFILES/TRA_HO_b"+str(b)+"X"+str(int(X))+"n"+str(nn)+"seed65438"
		if not os.path.isfile(trafile): trafile = "C:/AsusWebStorage/MySyncData/TRAFILES/TRA_HO_b"+str(b)+"X"+str(int(X))+"n"+str(n)+"seed65438"
		outfile = "./dat_LE_stream/HO/"+os.path.split(trafile)[1][4:]
		# trafile = "./dat_LE_stream/TRA_HO_b"+str(b)+"X"+str(int(X))+"n"+str(nn)+"seed65438"
		# if not os.path.isfile(trafile): trafile = "./dat_LE_stream/TRA_HO_b"+str(b)+"X"+str(int(X))+"n"+str(n)+"seed65438"
		# outfile = os.path.split(trafile)[0]+"/HO/"+os.path.split(trafile)[1][4:]
	rndfile = os.path.split(outfile)[0]+"/RND_"+os.path.split(outfile)[1]+".png"
	pdffile = os.path.split(outfile)[0]+"/PDF_"+os.path.split(outfile)[1]
	strfile = os.path.split(outfile)[0]+"/STR_"+os.path.split(outfile)[1]
	return trafile,rndfile,pdffile,strfile,n	
	
##================================================

def calc_grid(xydata, b,X, strfile, BW):
	""" Caluclate currents from time-series and interpolate onto grid """
	
	me = "LE_Plot.calc_grid: "
		
	## Output options
	fixscale = False	## If True, user determines axis scale
	savedata = True
	if fixscale: outfile = outfile+"_fix"
	
	## Set eta (yy) and xHO/xBW (x1)
	x1, yy = xydata
	del xydata
	
	## Set up grid of points in x-y
	gridsize = 30	
	if fixscale:	xmax, ymax = 2*X, blim(b,X)[1]
	else:			xmax, ymax = x1.max(), yy.max()
	x = np.linspace(-xmax,xmax, gridsize);y = np.linspace(-ymax,ymax,gridsize)
	xi,yi = np.meshgrid(x,y); yi = yi[::-1,:]	## Need to flip yi
	
	## Calculate speeds (1D arrays)
	vx1 = np.gradient(x1)
	vyy = np.gradient(yy)
	
	## --------------------------------------------------------------------
	## Interpolate data onto grid	
	t0 = time.time()
	
	## Scipy griddata (slow)
	gvx11 = griddata(zip(x1,yy), vx1, (xi,yi), method='linear',fill_value=0.0)
	gvyy1 = griddata(zip(x1,yy), vyy, (xi,yi), method='linear',fill_value=0.0)
	# gv1   = np.sqrt(gvx11*gvx11+gvyy1*gvyy1)
	print me+"Gridding data ",round(time.time()-t0,1),"seconds"
	
	"""## Split up triangulation step and interpolation step
	## gridpoints = np.array([[i,j] for i in y for j in x])
	## Reminder: (x,y)->(row,col), so indices must be reversed"""
	# vertices,weights = interp_weights(np.array(zip(x1,yy)), np.array([[i,j] for i in y for j in x]))
	# print me+"Triangulation",round(time.time()-t0,1),"seconds"; t1=time.time()
	# gvx11 = interpolate(vx1, vertices, weights).reshape([gridsize,gridsize])
	# gvyy1 = interpolate(vyy, vertices, weights).reshape([gridsize,gridsize])
	# gv1   = interpolate(v1,  vertices, weights).reshape([gridsize,gridsize])
	# print me+"Interpolation",round(time.time()-t1,1),"seconds"; t1=time.time()
	
	## Write data file and header file
	if savedata:
		LE_Simulate.save_data(strfile, np.vstack([x,y,gvx11,gvyy1]) )
		np.savetxt(strfile+".hdr",np.array([b,X,xmax,ymax,BW]) )
	
	return x,y,gvx11,gvyy1,(b,X,xmax,ymax,BW)
	

def plot_stream(x,y,gvx,gvy,oargs,outfile):
	""" From grid coordinates and values, plot streamlines """
	
	me = "LE_Plot.plot_stream: "
	
	## Expand out parameters
	b,X,xmax,ymax,BW,smooth = oargs
	gv = np.sqrt(gvx*gvx+gvy*gvy)
		
	showplot = False

	## Smooth data
	if smooth is not 0.0:
		gvy = gaussian_filter(gvy, smooth)
		gvx = gaussian_filter(gvx, smooth)
		gv  = gaussian_filter(gv, smooth)
	outfile += "_sm"+str(smooth)
		
	## --------------------------------------------------------------------	
	
	## Plotting
	
	t0 = time.time()
	fs = 25
	
	fig = plt.figure(facecolor="white")
	fig.suptitle(outfile)
	
	## Add subplot with exact solution
	if not BW:
		from LE_ExactHO import main as plot_exact
		ax1 = fig.add_subplot(121,aspect="auto")
		ax2 = fig.add_subplot(122,aspect="auto",sharey=ax1)
		plot_exact((ax2,xmax,ymax,b,False))
		fig.tight_layout();fig.subplots_adjust(top=0.93)
		print me+"Plotting exact",round(time.time()-t0,1),"seconds"
	else:
		ax1 = fig.add_subplot(111)
	
	## Accoutrements	
	ax1.set_xlim([-xmax,xmax]);	ax1.set_ylim([-ymax,ymax])
	ax1.set_xlabel("$x$",fontsize=fs);ax1.set_ylabel("$\eta$",fontsize=fs)
	## Plot wall positions if BW; plot separatrix if HO
	if BW:	plot_walls(ax1, X, xmax,ymax,2)
	else:	plot_separatrix(ax1, b, xmax, ymax, 2)
	
	## Line widths
	lw1 = 3.0*gv/gv.max()
	
	t0=time.time()
	## Plot absolute speed contour and streamplot
	## NOTE fudge (-) to force agreement with exact
	# ax1.contourf(xi,yi,gv, 4, alpha=0.4)
	ax1.streamplot(-x,y, -gvx,gvy, arrowsize=1.8, arrowstyle="->", linewidth=lw1, minlength=xmax/20)
			
	print me+"Plotting data ",round(time.time()-t0,1),"seconds"; t0=time.time()
		
	## Output
	fig.savefig(outfile+".png",facecolor=fig.get_facecolor(), edgecolor="none")
	print me+"Plot saved",outfile+".png"
	if showplot:	plt.show()

	plt.close()
			
	return

##================================================

def plot_stream_from_file(strfile,smooth=1.0):
	"""
	PURPOSE
	From a pre-gridded STR_....npy file, plot streamlines.
	
	EXAMPLE
	>>> LE_Plot.plot_stream_from_file("./dat_LE_stream/b=0.1/STR_BW_b0.1X1n2000000seed65438",0.5)
	"""
	me = "LE_Plot.plot_stream_from_file: "
	# if strfile is None: strfile = argv[1]
	try:
		A = np.load(strfile+".npy"); grd = A.shape[1]
		x,y,gvx,gvy = A[0],A[1],A[2:grd+2],A[grd+2:]
		oargs = np.loadtxt(strfile+".hdr")
		print me+"File found",strfile+".npy"
	except IOError:
		raise IOError(me+"File\n "+strfile+".npy\n not found. Abort.") 
	plot_stream( x,y,gvx,gvy, np.append(oargs,smooth), strfile )
	return	
	
##================================================




##================================================

def plot_rand(txyxidata, b,X, outfile):
	""" Plot stochastic forces and response of x """
	
	me = "LE_Plot.plot_rand: "
	if os.path.isfile(outfile): return me+"skip"
	t0 = time.time()
	showplot = False
	
	t, x, eta, xi = txyxidata
	del txyxidata
	tmax = np.ceil(t.max())
	
	## Plot walk
	fs = 25
	winsize = int(tmax/80)
	fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)
	fig.suptitle(outfile)#+"\n"+str(argv)[1:-1])
	envelope_plot(t, xi, winsize, ax=ax1)
	ax1.set_ylabel("$\\xi$",fontsize=fs)
	envelope_plot(t, eta, winsize, ax=ax2)
	ax2.set_ylabel("$\eta$",fontsize=fs)
	envelope_plot(t, x, winsize, ax=ax3)
	ax3.plot([0,t.max()],[X,X],"k--"); ax3.plot([0,t.max()],[-X,-X],"k--")
	ax3.set_xlabel("$t$",fontsize=fs);ax3.set_ylabel("$x$",fontsize=fs)
	etalim = np.ceil(abs(eta).max())	## Not perfect
	#fig.tight_layout()
	plt.savefig(outfile)
	print me+"Plot saved as",outfile
	print me+"Plotting random data:",round(time.time()-t0,1),"seconds"
	if showplot:		plt.show()	
	
	plt.close(fig)	
	return
	
##================================================

def plot_pdf(data,b,X,outfile):
	""" Plot pdf of positions in x-eta space """	
	me = "LE_Plot.plot_pdf: "
	showplot = False
	t0 = time.time()
	## Data
	x, y = data
	xmax, ymax = np.abs(x).max(), np.abs(y).max()
	## Plot pdf
	fs = 25
	# counts, xedges, yedges, im = plt.hist2d(x,y, bins=100, range=[[-2*X,+2*X],blim(b,X)], normed=True)
	counts, xedges, yedges, im = plt.hist2d(x,y, bins=100, range=[[-xmax,+xmax],[-ymax,ymax]], normed=True)
	plt.xlabel("$x$",fontsize=fs);plt.ylabel("$\eta$",fontsize=fs)
	plt.suptitle(outfile)
	plt.savefig(outfile+".png")
	## Output
	print me+"Plot saved as",outfile+".png"
	if showplot:	plt.show()
	plt.close()		
	print me+"Plotting PDF:",round(time.time()-t0,1),"seconds"
	return counts.T, xedges, yedges

##================================================

def griddata_CS(coords,values,grid):
	"""
	Modification to scipy.interpolate.griddata which reduces redundancy
	http://stackoverflow.com/questions/20915502/speedup-scipy-griddata-for-multiple-interpolations-between-two-irregular-grids
	"""
	coords = np.array(coords)
	vertices,weights = interp_weights(coords, grid)
	g_data = interpolate(values, vertices, weights)
	return g_data

def interp_weights(xyz, uvw):
	"""
	Returns vertices of triangulation and weights to apply to values in each cell.
	"""
	
	## Dimension of data
	d = xyz.shape[-1]
	
	## Delaunay triangulation of random points
	tri = sp.spatial.Delaunay(xyz)
	
	## Find indices of triangles approximating regular grid
	## -1 if grid point is outside tesselated region (N)
	simplex = tri.find_simplex(uvw)
	
	## Grid-triangles' vertex indices (N,3)
	## tri.simplices returns indices of all triangles in tesselation
	vertices = np.take(tri.simplices, simplex, axis=0)
	
	## The transformed coordinates of grid-triangles
	## tri.transform.shape = (xyz.shape[0], 3, d)
	## temp.shape = (uvw.shape[0], 3, d)
	## The last vertex-coordinate for each row temp[:,-1,:]=(0,0) roughly
	temp = np.take(tri.transform, simplex, axis=0)

	## Gridpoint (uvw) coordinates caluclated from triangulation-coordinate zeros
	delta = uvw - temp[:, d, :]
	
	## For each gridpoint in uvw, dot coordinate-offset delta=(dx,dy) with
	##  the coordinate of the corresponding triangle's vertices temp[:d]=((v1x,v1y),(v2x,v2y))
	## Result is ((v1x*dx+v1y*dy),(v2x*dx+v2y*dy)) for each row (each gridpoint) in uvw
	## bary.shape = (uvw.shape[0], d)
	bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
	
	## Return (i) indices of triangles corresponding to uvw gridpoints;
	## (ii)
	return vertices, np.hstack([bary, 1 - bary.sum(axis=1, keepdims=True)])

def interpolate(values, vtx, wts, fill_value=0.0):

	## Matrix element-wise multiplication
	ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
	
	## Some gridpoints outside random-points domain
	ret[np.any(wts < 0, axis=1)] = fill_value
	
	return ret



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

def blim(b,X):
	if 1:# X >= 10:
		return [-1,1]#[-0.5,0.5]
	else:
		return [-2*b*X,2*b*X]

##==========================================

def plot_separatrix(ax, b, xmax,ymax, lw=2):
	"""
	Affix theoretical separatrix to plot.
	Messy.
	"""
	eigx = 1/np.sqrt(b*(b+1))*ymax
	ax.plot([-eigx,eigx],[-ymax,ymax],"k--",linewidth=lw)
	ax.plot([eigx,-eigx],[-ymax,ymax],"k--",linewidth=lw)
	return
	
def plot_walls(ax, X, xmax,ymax, lw=2):
	"""
	Plot potential walls
	"""
	ax.plot([-X,-X],[-ymax,ymax],"r--",[X,X],[-ymax,ymax],"r--",linewidth=2)
	return
	
	
##================================================	

def LEpdf(xydata):
	""" Plot pdf 
	NEEDS WORK
	"""
	
	## Read eta (yy), xHO (x1) points from file
	yy,x1 = np.loadtxt(xydata,delimiter=" ",skiprows=1)[:,1:3].T
	del xydata

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
def histogram_data(xydata):
	"""
	Homegrown.
	Construct a histogram from a large CSV file by iterating.
	"""

	with open(xydata, "rb") as csvfile:
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
