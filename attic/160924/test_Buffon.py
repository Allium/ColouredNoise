import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt

# How many hits?
# Hits per length of boundary?
# Angle of hits?
# All as a function of l


def main():

	l = 0.3
	R = 1.0
	S = 0.5

	## Wall
	ang = np.linspace(0.0,2*np.pi,36)
	plt.plot(R*np.cos(ang),R*np.sin(ang))
	plt.plot(S*np.cos(ang),S*np.sin(ang))

	## Needle centre, angle and orientation
	N = 50
	r = np.random.triangular(0.0,R,R,N)
	r = r[r>S]
	# r  = r[r>=R-l]	## Show only needles with a chance of intersecting
	th = 2*np.pi*np.random.random(N)[r>S]
	ph = 2*np.pi*np.random.random(N)[r>S]

	## Needle endpoints
	needs = np.array([[r*np.cos(th),r*np.sin(th)],\
				[r*np.cos(th)+l*np.cos(ph),r*np.sin(th)+l*np.sin(ph)]]).T
				
	## Outermost
	rout = np.sqrt((needs[:,:,1]*needs[:,:,1]).sum(axis=-1))
	Rintidx = (rout>=R); Sintidx = (rout<=S)
	intidx = Sintidx+Rintidx
	intneeds = needs[intidx]
	
	print Rintidx.sum(), Sintidx.sum()
	print Rintidx.sum()/R, Sintidx.sum()/S

	plt.xlim([-R-l,R+l]); plt.ylim([-R-l,R+l])

	## Plot needles
	for i in range(needs.shape[0]):
		plt.plot(needs[i,0],needs[i,1], alpha=0.2)
	for i in range(intneeds.shape[0]):
		plt.plot(intneeds[i,0],intneeds[i,1])
	plt.plot(needs[:,0,0],needs[:,1,0],"ro")

	## Find intersection
	# beta = np.zeros(intidx.sum())
	ri, thi, phi = r[intidx], th[intidx], ph[intidx]
	needi = needs[intidx]
	for i in range(ri.shape[0]):
		L = R if Rintidx[intidx][i] else S
		aa = thi[i]
		for j in range(10):	aa = asol(aa, ri[i], thi[i], phi[i], L)
		ll = np.sqrt(ri[i]*ri[i]+L*L-2*ri[i]*L*np.cos(thi[i]-aa))
		# beta[i] = aa+phi[i]
		plt.plot(needi[i,0,0]+ll*np.cos(phi[i]), needi[i,1,0]+ll*np.sin(phi[i]), "kx")
	
	plt.show()
	return


def eqs(coord):
	ri, thi, phi, ll, aa = coord
	return (ri*ri+ll*ll+ri*ll*np.cos(thi+phi)-R*R,\
			np.arctan2(ri*np.sin(thi)+ll*np.sin(phi),ri*np.cos(thi)+ll*np.cos(phi))-aa)

def asol(a, r, th, ph, S):
	b = np.sqrt(r*r+S*S-2*r*S*np.cos(th-a))
	return np.arctan2(r*np.sin(th)+b*np.sin(ph),r*np.cos(th)+b*np.cos(ph))

				
if __name__=="__main__":	main()				
				