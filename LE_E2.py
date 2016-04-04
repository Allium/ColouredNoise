
import numpy as np
from matplotlib import pyplot as plt
import os, optparse, glob
from LE_Utils import filename_pars


def main():
	
	me = "LE_E2: "
	
	parser = optparse.OptionParser(conflict_handler="resolve")
	parser.add_option('-a','--alpha',
		dest="alpha", default=-1.0, type="float")
	parser.add_option('-s','--show',
		dest="showfig", default=False, action="store_true")
	parser.add_option('-v','--verbose',
		dest="verbose", default=False, action="store_true")
	opt, args = parser.parse_args()
	dirpath = args[0]
	a = opt.alpha
	showfig = opt.showfig
	vb = opt.verbose
	
	assert os.path.isdir(dirpath), me+"Need directory as argument."
	assert a >= 0.0, me+"Must specify alpha."
	dirpars = filename_pars(dirpath)
	assert dirpars["ftype"] is "linear", me+"Check input."
	
	## 
	if dirpars["geo"] is "CIR":
		plotfile = dirpath+"E2_CIRLPDF_a"+str(a)+".png"
		histfiles = sort_AN(glob.glob(dirpath+"/BHIS*_a"+str(a)+"*.npy"), "R")[::-1]
	elif dirpars["geo"] is "1D":
		plotfile = dirpath+"E2_1DLPDF_a"+str(a)+".png"
		histfiles = sort_AN(glob.glob(dirpath+"/BHIS*_a"+str(a)+"*.npy"), "X")[::-1]
	
	plot_pdf(histfiles,a,vb)
	
	plt.savefig(plotfile)
	if vb:	print me+"Figure saved to",plotfile
	if showfig:	plt.show()
	
	return

##=============================================================================	
def plot_pdf(histfiles,a,vb):

	geo = filename_pars(histfiles[0])["geo"]
	[wall,xbin,ebin] = ["R","rbins","erbins"] if geo is "CIR" else ["X","xbins","ybins"]
	## Note that everything is written in language of r

	fig, axs = plt.subplots((len(histfiles)+1)/2+1,2,sharey=True)
	axs = np.ravel(axs)
	
	for i,hf in enumerate(histfiles):
		
		R = filename_pars(hf)[wall]
		
		H = np.load(hf)
		if geo is "CIR": H = H.T[::-1]
		H[:,0]=H[:,1]
		bins = np.load(os.path.dirname(hf)+"/BHISBIN"+os.path.basename(hf)[4:-4]+".npz")
		rbins  = bins[xbin]
		erbins = bins[ebin]
		r = 0.5*(rbins[1:]+rbins[:-1])
		er = 0.5*(erbins[1:]+erbins[:-1])
		
		if geo is "CIR": H /= np.multiply(*np.meshgrid(r,er))
		
		ax = axs[i]
		ax.imshow(H, extent=[rbins[0],rbins[-1],erbins[0],erbins[-1]],aspect="auto")
		ax.axvline(R,c="k")
		ax.set_xlim(left=0.0)
		ax.set_ylim(bottom=0.0)
		ax.set_title("$"+wall+" = "+str(R)+"$")
	
	rbins = np.linspace(0.0,rbins[-1],100)
	erbins = np.linspace(0.0,erbins[-1],100)
	r = 0.5*(rbins[1:]+rbins[:-1])
	er = 0.5*(erbins[1:]+erbins[:-1])
	
	ax = axs[-2]
	ax.imshow(pdf_E2(r,er,a), extent=[rbins[0],rbins[-1],erbins[0],erbins[-1]],aspect="auto")
	ax.set_title("E2 $"+wall+" = 0$")
	
	ax = axs[-1]
	R, ER = np.meshgrid(r,er); ER = ER[::-1]
	ax.imshow(np.exp(-0.5*R*R-0.5*ER*ER), extent=[rbins[0],rbins[-1],erbins[0],erbins[-1]],aspect="auto")
	ax.set_title("Uncorrelated")
	
	plt.tight_layout()
				
	return None
	
	
def pdf_E2(x,eta,a):
	X, ETA = np.meshgrid(x,eta)
	ETA = ETA[::-1]
	rho0 = 2*np.pi / (np.sqrt(a)*(a+1))
	rho = np.exp(-0.5*(a+1)**2*X*X-0.5*a*(a+1)*ETA*ETA+a*(a+1)*X*ETA)
	return rho0 * rho
	
def sort_AN(filelist, var):
	varlist = [filename_pars(filename)[var] for filename in filelist]
	idxs = np.argsort(varlist)
	return [filelist[idx] for idx in idxs]
		
	
if __name__=="__main__":
	main()