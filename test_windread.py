me= "test_windread"

"""
Add new data to existing WIND file.
"""

import numpy as np
import os
from sys import argv

windfile = "Pressure/RECT/RECT_CAR_RL_t20.0/originalWIND/WIND_u1.0.npz"

data = np.load(windfile)
print me+"Data file found:",windfile
A, L, W = data["A"], data["L"], data["W"]
del data


## Add rows
numL = np.unique(L).size
## a=0.02
A = np.hstack([[0.02]*numL,A])
L = np.hstack([L[:numL],L])
W = np.vstack([[-0.57,-0.175,-0.0531,-0.00909,-0.0009],W])	## dt=0.05

## Modify existing rows
### a=0.1. From t=150 dt=0.05
#idx = np.nonzero((A==0.1))[0]
#W[divmod(idx,numL)[0],:]=[1.17e-5,1.51e-3,9.18e-4,1.37e-4,1.95e-5]

### Modify values
### a=0.2, l=0.4. From t=300 dt=0.01.
#idx = np.nonzero((A==0.2)*(L==0.6*5))[0]
#W[divmod(idx,numL)[0],divmod(idx,numL)[1]]=1.29e-3
### a=0.1, l=0.4. From t=400 dt=0.005.
#idx = np.nonzero((A==0.1)*(L==0.6*5))[0]
#W[divmod(idx,numL)[0],divmod(idx,numL)[1]]=1.43e-3
### a=0.05, l=0.4. From t=400 dt=0.005.
#idx = np.nonzero((A==0.1)*(L==0.6*5))[0]
#W[divmod(idx,numL)[0],divmod(idx,numL)[1]]=1.43e-3

windfile_new = os.path.dirname(windfile)+"/../"+os.path.basename(windfile)
np.savez(windfile_new, A=A, L=L, W=W)
print me+"Update saved to:",windfile_new
