me0 = "test_etaPDFscale"

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from sys import argv

fsa, fsl, fst = 16, 12, 16

"""
Plot width and area of eta pdf (assumed Gaussian) against alpha for various annulus parameters.
Data from 161008 (finite) and 161011 (zero). Extraction script at bottom.
"""

try: nosave = not bool(argv[1])
except IndexError: nosave = True

plotscale = False	## Plot s2 or A
ZB = False			## Plot zero-bulk or finite-bulk data

a = np.array([0.2,0.5,1.0,2.0,3.0,5.0,10.0])
R = np.array([0.0,5.0,10.0,15.0,20.0])
S = np.array([0.0,2.0,5.0,9.0,10.0])

## Area and width for ZERO BULK. Dimensions a.size*R.size*4
s2_ZB = np.array([
		[[ 5.257,  0. ,    5.257,  0.   ],
		 [ 5.25 ,  0. ,    5.248,  5.257],
		 [ 5.255,  0. ,    5.255,  5.258],
		 [ 5.251,  0. ,    5.249,  5.256],
		 [ 5.255,  0. ,    5.253,  5.261]],
		[[ 2.04 ,  0. ,    2.04 ,  0.   ],
		 [ 2.044,  0. ,    2.049,  2.041],
		 [ 2.043,  0.  ,   2.043,  2.046],
		 [ 2.042,  0. ,    2.043,  2.043],
		 [ 2.039,  0.  ,   2.04 ,  2.041]],
		[[ 1.009,  0.  ,   1.009,  0.   ],
		 [ 1.008,  0.  ,   1.015,  1.003],
		 [ 1.01 ,  0. ,    1.013,  1.009],
		 [ 1.009,  0.  ,   1.011,  1.011],
		 [ 1.009,  0. ,    1.013,  1.007]],
		[[ 0.508,  0.  ,   0.508,  0.   ],
		 [ 0.51 ,  0.  ,   0.523,  0.494],
		 [ 0.509,  0.  ,   0.513,  0.507],
		 [ 0.506,  0.  ,   0.509,  0.506],
		 [ 0.507,  0.  ,   0.513,  0.504]],
		[[ 0.334,  0.  ,   0.334,  0.   ],
		 [ 0.335,  0.  ,   0.35 ,  0.316],
		 [ 0.336,  0.  ,   0.341,  0.334],
		 [ 0.336,  0.  ,   0.339,  0.336],
		 [ 0.334,  0.  ,   0.341,  0.328]],
		[[ 0.2  ,  0.  ,   0.2  ,  0. ],
		 [ 0.201,  0.  ,   0.216,  0.181],
		 [ 0.201,  0.  ,   0.21 ,  0.193],
		 [ 0.2  ,  0.  ,   0.206,  0.198],
		 [ 0.2  ,  0.  ,   0.204,  0.2  ]],
		[[ 0.1  ,  0.  ,   0.1  , 0. ],
		 [ 0.1  ,  0. ,    0.114,  0.082],
		 [ 0.099,  0.  ,   0.105 , 0.096],
		 [ 0.1  ,  0.  ,   0.108 , 0.093],
		 [ 0.1  ,  0.  ,   0.104 , 0.1  ]]])
A_ZB  = np.array([
	[[ 1.001,  0.,    1.001,  0.   ],
	[ 1.001 , 0. ,    0.591,  0.407],
	[ 1.001 , 0. ,    0.546,  0.451],
	[ 1.001 , 0. ,    0.532,  0.466],
	[ 1.001 , 0. ,    0.525,  0.473]],
	[[ 1.,  0.,  1.,  0.],
	[ 1.001 , 0.  ,   0.601,  0.397],
	[ 1.001 , 0.  ,   0.551,  0.447],
	[ 1.001 , 0.  ,   0.535,  0.463],
	[ 1.    , 0.  ,   0.525,  0.472]],
	[[ 1.,  0.,  1.,  0.],
	[ 1.    , 0. ,    0.611,  0.386],
	[ 1.    , 0. ,    0.558,  0.439],
	[ 1.    , 0. ,    0.539,  0.457],
	[ 1.001 , 0. ,    0.531,  0.466]],
	[[ 1.001,  0.,     1.001,  0.   ],
	[ 1.003 , 0. ,    0.633,  0.366],
	[ 1.001 , 0. ,    0.569,  0.427],
	[ 1.001 , 0. ,    0.55 ,  0.447],
	[ 1.    , 0. ,    0.537,  0.459]],
	[[ 1.,  0.,  1.,  0.],
	[ 1.001 , 0. ,    0.648,  0.349],
	[ 0.999 , 0. ,    0.578,  0.417],
	[ 1.002 , 0. ,    0.556,  0.442],
	[ 1.002 , 0. ,    0.543,  0.455]],
	[[ 1.,  0.,  1.,  0.],
	[ 1.001 , 0. ,    0.672,  0.328],
	[ 1.001 , 0. ,    0.596,  0.401],
	[ 1.    , 0. ,    0.566,  0.43 ],
	[ 1.    , 0. ,    0.552,  0.443]],
	[[ 1.002,  0.,     1.002,  0.   ],
	[ 1.001 , 0. ,    0.717,  0.289],
	[ 1.    , 0. ,    0.624,  0.374],
	[ 0.999 , 0. ,    0.588,  0.407],
	[ 1.    , 0. ,    0.569,  0.427]]])

		
## Area and width for FINITE BULK. Dimensions a.size*S.size*4
s2_FB = np.array([
	[[ 5.253,  5.247,  5.274,  0.   ],
	[ 5.256 , 5.249 , 5.275 , 5.286],
	[ 5.256 , 5.243 , 5.275 , 5.291],
	[ 5.248 , 5.178 , 5.274 , 5.28 ],
	[ 5.255 , 0.    , 5.255 , 5.258]],
	[[ 2.037,  2.026,  2.073,  0.   ],
	[ 2.04  , 2.027 , 2.075 , 2.096],
	[ 2.038 , 2.018 , 2.077 , 2.077],
	[ 2.043 , 1.964 , 2.073 , 2.085],
	[ 2.042 , 0.    , 2.043 , 2.043]],
	[[ 1.01 ,  0.998,  1.047,  0.   ],
	[ 1.01  , 0.993 , 1.055 , 1.063],
	[ 1.01  , 0.987 , 1.048 , 1.064],
	[ 1.009 , 0.929 , 1.046 , 1.049],
	[ 1.009 , 0.    , 1.013 , 1.007]],
	[[ 0.508,  0.492,  0.554,  0.   ],
	[ 0.51  , 0.492 , 0.555 , 0.567],
	[ 0.509 , 0.48  , 0.559 , 0.56 ],
	[ 0.506 , 0.427 , 0.55  , 0.549],
	[ 0.507 , 0.    , 0.513 , 0.504]],
	[[ 0.335,  0.317,  0.381,  0.   ],
	[ 0.335 , 0.317 , 0.381 , 0.383],
	[ 0.335 , 0.309 , 0.382 , 0.384],
	[ 0.334 , 0.265 , 0.375 , 0.377],
	[ 0.334 , 0.    , 0.341 , 0.328]],
	[[ 0.2  ,  0.183,  0.244,  0.   ],
	[ 0.201 , 0.182 , 0.245 , 0.247],
	[ 0.2   , 0.175 , 0.244 , 0.247],
	[ 0.201 , 0.145 , 0.239 , 0.237],
	[ 0.201 , 0.    , 0.21  , 0.193]],
	[[ 0.1  ,  0.084,  0.136,  0.   ],
	[ 0.1   , 0.084 , 0.136 , 0.134],
	[ 0.101 , 0.08  , 0.136 , 0.137],
	[ 0.1   , 0.062 , 0.132 , 0.127],
	[ 0.1   , 0.    , 0.108 , 0.093]]])
A_FB  = np.array([
	 [[ 1.001,  0.767,  0.232,  0.   ],
	 [ 1.001 , 0.742 , 0.234 , 0.024],
	 [ 1.001 , 0.644 , 0.258 , 0.096],
	 [ 1.    , 0.273 , 0.417 , 0.307],
	 [ 1.001 , 0.    , 0.546 , 0.451]],
	 [[ 1.   ,  0.754,  0.245,  0.   ],
	 [ 1.    , 0.73  , 0.246 , 0.023],
	 [ 1.    , 0.632 , 0.271 , 0.095],
	 [ 1.001 , 0.275 , 0.423 , 0.3  ],
	 [ 1.001 , 0.    , 0.551 , 0.447]],
	 [[ 1.   ,  0.735,  0.264,  0.   ],
	 [ 1.001 , 0.711 , 0.266 , 0.023],
	 [ 1.    , 0.615 , 0.288 , 0.095],
	 [ 1.001 , 0.277 , 0.429 , 0.292],
	 [ 1.    , 0.    , 0.558 , 0.439]],
	 [[ 1.001,  0.702,  0.299,  0.   ],
	 [ 1.002 , 0.68  , 0.301 , 0.02 ],
	 [ 1.001 , 0.59  , 0.318  ,0.093],
	 [ 1.    , 0.276 , 0.445  ,0.28 ],
	 [ 1.001 , 0.    , 0.569  ,0.427]],
	 [[ 1.   ,  0.676,  0.326,  0.   ],
	 [ 1.002 , 0.656 , 0.326 , 0.02 ],
	 [ 1.001 , 0.572 , 0.341 , 0.09 ],
	 [ 1.    , 0.276 , 0.456 , 0.273],
	 [ 0.999 , 0.    , 0.578 , 0.417]],
	 [[ 1.   ,  0.638 , 0.366,  0.   ],
	 [ 1.    , 0.619 , 0.367 , 0.018],
	 [ 1.001 , 0.543 , 0.377 , 0.087],
	 [ 1.001 , 0.269 , 0.479 , 0.264],
	 [ 1.001 , 0.    , 0.596 , 0.401]],
	 [[ 0.998,  0.573,  0.436,  0.   ],
	 [ 1.    , 0.557 , 0.439 , 0.016],
	 [ 1.001 , 0.49  , 0.446 , 0.079],
	 [ 1.001 , 0.252 , 0.528 , 0.243],
	 [ 1.    , 0.    , 0.624 , 0.374]]])

ylabel = r"$\alpha\cdot\sigma^2$" if plotscale else r"$A$"

if ZB:
	Y = a[:,np.newaxis,np.newaxis]*s2_ZB if plotscale else A_ZB
	Z = R
	tit = r"$R=S$"
else:
	Y = a[:,np.newaxis,np.newaxis]*s2_FB if plotscale else A_FB
	Z = S
	tit = r"$R=10$"

subtit = ["Total","Bulk","Outer","Inner"]
	
	
fig, axs = plt.subplots(2,2, sharex=True,sharey=True)
axs = axs.flatten()

for i, ax in enumerate(axs):
	## To plot against alpha
	for j, z in enumerate(Z):
		ax.plot(a, Y[:,j,i], "o-", label=r"$S=%.1f"%(z)+"$")
	# ## To plot against R or S
	# for j, aj in enumerate(a):
		# ax.plot(Z, Y[j,:,i], "o-", label=r"$\alpha=%.1f"%(aj)+"$")

for i, ax in enumerate(axs):
	ax.set_title(subtit[i])
	ax.grid()
		
axs[2].set_xlabel(r"$\alpha$", fontsize=fsa)
axs[3].set_xlabel(r"$\alpha$", fontsize=fsa)
axs[0].set_ylabel(ylabel, fontsize=fsa)
axs[2].set_ylabel(ylabel, fontsize=fsa)

axs[0].legend(loc="best", fontsize=fsl)	

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle(tit, fontsize=fst)	

if not nosave:
	filestr = ["_ZB" if ZB else "_FB", "_s2" if plotscale else "_A"]
	plotfile = "etaPDF"+filestr[1]+filestr[0]+".jpg"
	fig.savefig(plotfile)
	print "Figure saved to",plotfile

plt.show()


## ============================================================================

exit()

## Data produced using following script

## Zero bulk
for a in [0.2,0.5,1.0,2.0,3.0,5.0,10.0]:
	for R in [0.0,5.0,15.0,20.0]:
		os.system("python test_etaPDF.py Pressure/161011_CIR_DL_dt0.01/BHIS_CIR_DL_a%.1f_R%.1f_S%.1f_dt0.01.npy"%(a,R,R))
exit()

## Finite bulk
for a in [0.2,0.5,1.0,2.0,3.0,5.0,10.0]:
	for S in [0.0,2.0,5.0,9.0,10.0]:
		os.system("python test_etaPDF.py Pressure/161008_CIR_DL_dt0.01/BHIS_CIR_DL_a%.1f_R10.0_S%.1f_dt0.01.npy"%(a,S))
exit()


