me0 = "GO"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import NullLocator, MaxNLocator
from sys import argv

from LE_Utils import fs, set_mplrc

## MPL defaults
set_mplrc(fs)

##=============================================================================

def main():
	"""
	First argument is alpha
	Second is ftype string "c" or "l"
	"""
	a = float(argv[1])
	ftype = argv[2]
	
	plot_trajectory(a, ftype)
	
	plt.show()
	
	return
	

##=============================================================================

def plot_trajectory(a, ftype):
	"""
	"""
	me = me0+".plot_trajectory: "
	
	E0max = 5.0
	E0min = 1.0 if ftype=="c" else 0.0
	
	NE0 = int((E0max-E0min)*3)+1
	Eta0 = np.linspace(E0min,E0max,NE0)
	
	fig, ax = plt.subplots(1,1, figsize=fs["figsize"])
	
	## Each trajectory has different initial value
	for eta0 in Eta0:
		N = int(eta0*100)+1		## Number of points
		eta = np.linspace(1.0/N,eta0,N)[::-1]
		x = calc_trajectory_C(a,eta) if ftype=="c" else calc_trajectory_L(a,eta)
		
		ax.plot(x,eta)
		
		## Arrows
		hw = 0.4 if ftype=="c" else 0.15
		i = 3 if ftype=="c" else np.log(2)
		j = int(i*eta.size/10)
		color = ax.lines[-1].get_color()
		try:
			ax.arrow(x[j], eta[j], x[j+1]-x[j], eta[j+1]-eta[j],
					shape='full', lw=0, length_includes_head=True, head_starts_at_zero=True,
					head_length=2*hw, head_width=hw, color=color, overhang=0.4)
		except IndexError:
			pass
		
	ax.set_xlim(left=0.0)
	ax.xaxis.set_major_locator(MaxNLocator((6 if ftype=="c" else 4)))
	ax.yaxis.set_major_locator(MaxNLocator(6))
	ax.set_xlabel(r"$x$")
	ax.set_ylabel(r"$\eta$")
	ax.grid()
	
	## Inset 	
	left, bottom, width, height = [0.60, 0.63, 0.27, 0.25] if ftype=="c" else [0.60, 0.18, 0.27, 0.25]
	axin = fig.add_axes([left, bottom, width, height])
	x = np.linspace(-1.0,1.0,51)
	U = x.copy() if ftype=="c" else 0.5*x*x
	U[x<=0.0] = 0.0
	axin.plot(x,U,"k-")
#	axin.set_xlim([-1.0,1.0])
	axin.set_xlabel(r"$x$")
	axin.set_ylabel(r"$U$")
	axin.set_xticks([0.0])
	axin.set_xticklabels(["0"])
	axin.yaxis.set_major_locator(NullLocator())
	
	return
	
	
	
##=============================================================================

def calc_trajectory_C(a,eta):
	"""
	Calculate trajectory for constant force.
	"""
	return a*(eta[0]-eta + np.log(eta/eta[0]))
	

def calc_trajectory_L(a,eta):
	"""
	Calculate trajectory for constant force.
	"""
	return a/(a-1)*eta*(1-(eta/eta[0])**a)
	
##=============================================================================

def calc_pdf_C(a,eta0):
	"""
	"""
	
	return
	
	
def calc_pdf_L(a,eta0):
	"""
	"""
	
	return
	
##=============================================================================
##=============================================================================
if __name__=="__main__":
	main()
