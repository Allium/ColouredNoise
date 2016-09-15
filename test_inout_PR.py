import numpy as np
import matplotlib.pyplot as plt

"""
Plot the final pressure for (a) disc bulk, outer wall (out); (b) disc wall, outer bulk (in).
Numbers from LE_SPressure.py and LE_inSPressure.py acting on files 13-14/09/2016.
"""

## Plot pressure calculated in the conventional way against R
if 0:
	RS = np.array([1,3,4,5,6,7,8,9,10,15])

	## From tfac=5 run
	Out5 = np.array([1.143805,1.249528,1.221696,1.159210,1.134256,1.116733,1.056654,1.115144,1.005819,1.056129])
	In5  = np.array([0.222833,0.949242,0.898994,0.649940,0.763776,0.835591,0.828380,0.867031,0.858350,0.894757])
	## From tfac=10 run
	Out10 = np.array([1.177994,1.224845,1.178758,1.159542,1.114457,1.122078,1.085151,1.080827,1.096486,1.053408])
	In10  = np.array([0.219303,0.648968,0.736829,0.800179,0.858218,0.855152,0.914543,0.924211,0.945878,0.948403])

	Out = 1./3.*Out5 + 2./3.*Out10
	In  = 1./3.*In5  + 2./3.*In10

	# plt.plot(RS,Out5,label="Out5")
	# plt.plot(RS,Out10,label="Out10")
	# plt.plot(RS,In5,label="In5")
	# plt.plot(RS,In10,label="In10")
	plt.plot(RS,Out,label="Disc bulk",linewidth=2.)
	plt.plot(RS,In,label="Disc wall",linewidth=2.)

	r = np.linspace(RS[0],RS[-1],100)
	plt.plot(r,1+1.0/r,"k-")
	plt.plot(r,1-1.0/r,"k-")
	plt.axhline(1.0,c="k",ls="--")
	plt.axvspan(0,4,color="red",alpha=0.05)

	plt.ylabel("Pressure on wall (normalised)",fontsize=16)
	plt.xlabel("$R$ or $S$",fontsize=16)

	plt.grid()
	plt.legend(fontsize=12)
	plt.show()
	
	
## Plot pressure calculated from bulk constant against alpha
else:
	a = np.array([0.2,0.4,0.5,0.7,1.0,1.5,2.0,2.5,3.0])

	Outa = np.array([])
	Ina  = np.array([])

	plt.plot(a,Outa,label="Disc bulk",linewidth=2.)
	plt.plot(a,Ina,label="Disc wall",linewidth=2.)

	aa = np.linspace(0,a[-1],100)
	plt.plot(aa,1.0/np.sqrt(1+aa),"k-")

	plt.ylabel("Pressure on wall (normalised)",fontsize=16)
	plt.xlabel("$\\alpha$",fontsize=16)

	plt.grid()
	plt.legend(fontsize=12)
	plt.show()