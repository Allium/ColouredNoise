
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
	assert (dirpars["ftype"] is "linear" and dirpars["geo"] is "CIR"), me+"Check input."
	
	plotfile = dirpath+"CIRLPDF_a"+str(a)+".png"
	
	histfiles = sort_AN(glob.glob(dirpath+"/BHIS*_a"+str(a)+"*.npy"))[::-1]
	plot_pdf(histfiles,a,vb)
	
	plt.savefig(plotfile)
	if vb:	print me+"Figure saved to",plotfile
	if showfig:	plt.show()
	
	return

	
def plot_pdf(histfiles,a,vb):

	fig, axs = plt.subplots((len(histfiles)+1)/2+1,2,sharey=True)
	axs = np.ravel(axs)
	
	for i,hf in enumerate(histfiles):
		
		R = filename_pars(hf)["R"]
		
		H = np.load(hf).T[::-1]
		bins = np.load(os.path.dirname(hf)+"/BHISBIN"+os.path.basename(hf)[4:-4]+".npz")
		rbins  = bins["rbins"]
		erbins = bins["erbins"]
		r = 0.5*(rbins[1:]+rbins[:-1])
		er = 0.5*(erbins[1:]+erbins[:-1])
		
		H /= np.multiply(*np.meshgrid(r,er))
		
		ax = axs[i]
		ax.imshow(H, extent=[rbins[0],rbins[-1],erbins[0],erbins[-1]],aspect=1)
		ax.axvline(R,c="k")
		ax.set_xlim(left=0.0)
		ax.set_ylim(bottom=0.0)
		ax.set_title("$R = "+str(R)+"$")
	
	rbins = np.linspace(0.0,rbins[-1],100)
	erbins = np.linspace(0.0,erbins[-1],100)
	r = 0.5*(rbins[1:]+rbins[:-1])
	er = 0.5*(erbins[1:]+erbins[:-1])
	
	ax = axs[-2]
	ax.imshow(pdf_E2(r,er,a), extent=[rbins[0],rbins[-1],erbins[0],erbins[-1]],aspect=1)
	ax.set_title("E2 R=0")
	
	ax = axs[-1]
	R, ER = np.meshgrid(r,er); ER = ER[::-1]
	ax.imshow(np.exp(-0.5*R*R-0.5*ER*ER), extent=[rbins[0],rbins[-1],erbins[0],erbins[-1]],aspect=1)
	ax.set_title("Uncorrelated")
	
	plt.tight_layout()
				
	return None
	
def pdf_E2(x,eta,a):
	X, ETA = np.meshgrid(x,eta)
	ETA = ETA[::-1]
	rho0 = 2*np.pi / (np.sqrt(a)*(a+1))
	rho = np.exp(-0.5*(a+1)**2*X*X-0.5*a*(a+1)*ETA*ETA+a*(a+1)*X*ETA)
	return rho0 * rho
	
def sort_AN(filelist):
	Rlist = [filename_pars(filename)["R"] for filename in filelist]
	idxs = np.argsort(Rlist)
	return [filelist[idx] for idx in idxs]

# def plot_all():
	# from sys import argv
	# for a in np.arange(0.0,1.5,0.2):
		# os.system("python LE_E2.py "+argv[1]+" -v -a "+str(a))
		
	
if __name__=="__main__":
	main()