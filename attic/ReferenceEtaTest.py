import numpy as np
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt

dt = 0.01
a = 0.01
assert 1/(a*a)>=2.0

## True eta
i = 100	## Index of tmax
j = np.arange(1,i+1,1)
exp = np.exp(dt/(a*a)*(j-i))
exp[:int(i-10*(a*a/dt))]=0.0	## Approximation
xi = 1/np.sqrt(dt)*np.random.normal(0,1,i)
## eta is convolution of xi and exp. Lose padded parts and reverse time.
eta = dt/(a*a)*fftconvolve(xi,exp,"full")[-i:][::-1]

## Reference eta
I = i*int(1/(a*a))	## Larger array for downsampling
J = np.arange(1,I+1,1)
EXP = np.exp(dt*(J-I))
EXP[:int(I-10/dt)]=0.0	## Approximation
XI = 1/np.sqrt(dt)*np.random.normal(0,1,I)
ETA = dt*fftconvolve(XI,EXP,"full")[-I:][::-1]

## Scale reference
cETA = 1/(a)*np.array([np.trapz(chunk,dx=dt) for chunk in np.array_split(ETA,i)])

plt.plot(j,eta,label="Actual")
plt.plot(j,cETA,label="Rescaled")

plt.legend()
plt.show()