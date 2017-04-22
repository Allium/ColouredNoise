me= "test_windread"

"""
Add new data to existing WIND file.
"""

import numpy as np
import os
from sys import argv

windfile = "Pressure/RECT/RECT_CAR_RL_t150.0/originalwind_lam5/WIND_lam5.0.npz"

data = np.load(windfile)
print me+"Data file found:",windfile
A, L, W = data["A"], data["L"], data["W"]
del data

## Add rows
numL = np.unique(L).size
## a=0.05
A = np.hstack([[0.05]*numL,A])
L = np.hstack([L[:numL],L])
#W = np.vstack([[0.00,1.00e-3,2.00e-3,8.67e-4,1.77e-4],W])	## dt=0.01
W = np.vstack([[4.57e-5,9.53e-4,1.27e-3,3.95e-4,4.30e-5],W])	## dt=0.05
## a=0.02
A = np.hstack([[0.02]*numL,A])
L = np.hstack([L[:numL],L])
#W = np.vstack([[0.00,6.96e-4,1.70e-3,3.73e-3,2.11e-3],W])	## dt=0.01
W = np.vstack([[5.96e-5,4.20e-4,1.67e-3,1.01e-3,2.90e-4],W])	## dt=0.05

## Modify existing rows
## a=0.1. From t=150 dt=0.05
idx = np.nonzero((A==0.1))[0]
W[divmod(idx,numL)[0],:]=[1.17e-5,1.51e-3,9.18e-4,1.37e-4,1.95e-5]

## Modify values
## a=0.2, l=0.4. From t=300 dt=0.01.
idx = np.nonzero((A==0.2)*(L==0.6*5))[0]
W[divmod(idx,numL)[0],divmod(idx,numL)[1]]=1.29e-3
## a=0.1, l=0.4. From t=400 dt=0.005.
idx = np.nonzero((A==0.1)*(L==0.6*5))[0]
W[divmod(idx,numL)[0],divmod(idx,numL)[1]]=1.43e-3
## a=0.05, l=0.4. From t=400 dt=0.005.
idx = np.nonzero((A==0.1)*(L==0.6*5))[0]
W[divmod(idx,numL)[0],divmod(idx,numL)[1]]=1.43e-3

windfile_new = os.path.dirname(windfile)+"/../"+os.path.basename(windfile)
np.savez(windfile_new, A=A, L=L, W=W)
print me+"Update saved to:",windfile_new
