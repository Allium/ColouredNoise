import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg as splinalg
import matplotlib.pyplot as plt
import time, copy
from sys import argv
from numpy import gradient as D


##================================================
def main():
	"""
	STARTED
	18/02/2015
	
	PURPOSE
	Numerically integrate the coloured-noise FPE in 1. B+W and @. HO potentials.
	
	EXECUTION
	python FPE_NumericalIntegrate.py <IC>
	ipython FPE_NumericalIntegrate.py --matplotlib

	BUGS / NOTES / TODO
	-- Space is offset by one. Affects J and p.
	-- estep unstable if y-drift expanded
	
	-- abrupt F -> instability
	"""
	
	plt.ion()
	
	t0 = time.time(); tevo=0
	IM =	0	## Integration method; 0=RK2, 1=CN
	PS =	1	## Potential setup; 0=harmonic, 1=bulk+walls, 2=diffusion
	try: 	IC = int(argv[1])
	except:	IC = 3

	## System / evolution parameters
	N = 100
	global dx; dx = 1.
	global dy; dy = 1.
	dt = 1.0/(N*N)
	nsteps = int(1/dt)
	if IM == 1: dt*=10; #nsteps*=2

	## Space
	J = -np.arange(-N/2+1,N/2+1,1)[:,np.newaxis]
	X = np.arange(-N/2,N/2+2,1)
	## Force
	f = 1.
	F = -f*X
	if PS == 1:
		i_w = N/10; x_w = i_w*dx
		F = np.zeros(N+2);F[:i_w+2]=f;F[-(i_w+2):]=-f;	#Problem with index?
	elif PS == 2:
		f=0.0
	
	## Output
	framerate = 200#min([nsteps/50,200])
	outfile = "./img_FPE_SS/FPE_PS"+str(PS)+"_f"+str(f)+"_n"+str(nsteps)+"_IM"+str(IM)

##---------------------------------------------------

	## Construct CN matrix
	if IM == 1:
		CN,CNp = make_CN_sp(N,dt,dx,dy,f,F,J,PS)
		CN = splinalg.aslinearoperator(CN)

##---------------------------------------------------

	## Initial condition
	sig = 0.05*N; tD0 = 0.5*sig*sig
	if IC == 3:	## 2D Gaussian
		p0 = pGaussD(0,tD0,X[1:-1],J,N)
	p=copy.copy(p0)
		
##---------------------------------------------------
	
	## Set up plots
	
	vmin,vmax = (np.min(p0),np.max(p0))
	fig, (ax1,ax2) = plt.subplots(1,2, facecolor='white')
	im1 = ax1.imshow(p0, vmin=vmin,vmax=vmax, interpolation="nearest")
	
	## Comparison
	if PS==2: ## Diffusion
		im2 = ax2.imshow(pGaussD(0,tD0,X[1:-1],J,N), vmin=vmin,vmax=vmax, interpolation="nearest")
		subtit = ["Numerical","Diffusion, exact"]
	else: ## HO SS
		## Covariance matrix
		C = np.sqrt(8.)*(f**2) * np.array([[(f+1)**2.,+0.5*(f+1)**(3./2.)],[+0.5*(f+1)**(3./2.),(f+1)]])
		im2 = ax2.imshow(Gauss2D(X[1:-1],J,C), vmin=vmin,vmax=vmax, interpolation="nearest")
		subtit = ["Numerical","HO SS"]
	if PS==1:
		pass
		
	[[ax.set_title(tit),ax.set_xlabel("$x$"),ax.set_ylabel("$\eta$")] for ax,tit in zip((ax1,ax2),subtit)]
	fig.colorbar(im1, ax=[ax1,ax2], orientation='horizontal')
	
	frame=0
	
##---------------------------------------------------

	## Evolution
	for n in range(nsteps):
		t=n*dt
		t1=time.time()
		
		## BCs
		p = apply_BCs(p,"outflow")		

		## Choose method:
		if IM==1:
			## CN # test: cgs, gmres, lgmres
			p, info = splinalg.lgmres(CNp, CN.matvec(p.flatten()))#, p.flatten()
			p	=	p.reshape([N,N])
		else:
			## RK2
			kp1 = dt*estep_unexp(p,J,F,PS)
			kp2 = dt*estep_unexp(p+0.5*kp1,J,F,PS)
			p += kp2
		
		tevo+=time.time()-t1
		
		## Plot
		if n%(framerate)==0:
			plt.pause(1e-6)
			im1.set_data(p); im1.set_clim([p.min(),p.max()])
			if PS==2:	## Diffusion
				res = p-pGaussD(t,tD0,X[1:-1],J,N)
				im2.set_data(res);	im2.set_clim([p.min(),p.max()])
			plt.draw()
			frame+=1
			print "Step",n,"Time",round(n*dt,2),"\t Normalisation",intarr(p,dx)

##---------------------------------------------------
	
	plt.savefig(outfile+".png")
	
	## Outinfo
	print "Evolution",round(tevo/nsteps,3),"seconds per timestep."
	
	return

##================================================
##================================================

## Evolution / Euler step

def estep(P,J,F):
	""" Expanded version -- unstable """
	DP  = D(P)
	PFX = P * D(F)[1:-1]
	FPX = F[1:-1] * DP[1]
	YPX = J * DP[1]
	#YPY = J * DP[0]
	Ydr = D(J*P)[0]
	PYY = D(DP[0])[0]
	## Sum to get RHS of FPE
	RHS = -PFX-FPX-YPX+Ydr+PYY
	## Eliminate small values
	RHS[abs(RHS)<pow(10,-15)] = 0
	return RHS
	
def estep_unexp(P,J,F,PS):
	""" Unexpanded version: best option so far """
	XYdr = 0.0
	if PS is not 2:
		Xdr = -D( (F[1:-1]+J)*P, dx)[1]
		Ydr = D(J*P, dy)[0]
		XYdr = Xdr+Ydr
	Ydi = D(D(P)[0])[0]
	return  XYdr + Ydi
			

##================================================

def apply_BCs(p,type="outflow",c=(0,0)):
	if type=="outflow":
		p[:,1]=p[:,2];p[:,-2]=p[:,-3];p[1,:]=p[2,:];p[-2,:]=p[-3,:]
		p[:,0]=p[:,1];p[:,-1]=p[:,-2];p[0,:]=p[1,:];p[-1,:]=p[-2,:]
	elif type=="periodic":
		p[:,0]=p[:,-1];p[:,-1]=p[:,0];p[0,:]=p[-1,:];p[-1,:]=p[0,:]
	elif type=="fixed":
		p[:,0]=c[1];p[:,-1]=c[1];p[0,:]=c[0];p[-1,:]=c[0]
	return p
		

##================================================	
	
## Integrate dD array
def intarr(arr,dx):
	for i in range(len(arr.shape)):
		arr = np.trapz(arr,dx=dx,axis=0)
	return round(arr,3)

##================================================

def make_CN_sp(N,dt,dx,dy,f,F,J,PS):
	"""
	Construct sparse matrices for the Crank-Nicolson method.
	CN multiplies P^n and CNp multiplies P^{n+1}.
	"""
	
	NN = N*N
	
	if PS is not 2:
		## Coefficients
		a = -0.5*dt*(1+2/(dy*dy)-0.5*f/dx*(F[2:]-F[:-2]))	## Should the 1 be there?
		b =  0.5*dt*(1/(dy*dy)+0.5*J/dy)
		c =  0.5*dt*(1/(dy*dy)-0.5*J/dy)
		d =  0.5*dt*(-1./(2*dx)*(f*F[1:-1,np.newaxis].repeat(N,axis=1).T + J.repeat(N,axis=1)))
		e = -d
		
		## Diagonals
		ad = a[:,np.newaxis].repeat(N,axis=1).T.flatten()
		bd = b.repeat(N)
		cd = c.repeat(N)
		dd = d.flatten()
		ed = e.flatten()
	
	## Pure diffusion
	else:
		ad = -0.5*dt*(2/(dy*dy)) * np.ones(NN)
		bd =  0.5*dt*(1/(dy*dy)) * np.ones(NN)
		cd =  0.5*dt*(1/(dy*dy)) * np.ones(NN)
		dd = np.zeros(NN)
		ed = np.zeros(NN)
				
	## Construct matrices
	dia = np.array([1+ad,bd,cd,dd,ed])
	diap= np.array([1-ad,-bd,-cd,-dd,-ed])
	off = np.array([0,+N,-N,+1,-1])
	
	CN = sp.sparse.dia_matrix((dia ,off),shape=(NN,NN))
	CNp= sp.sparse.dia_matrix((diap,off),shape=(NN,NN))
	
	return CN,CNp

##================================================

def pGaussD(t,t0,x,y,N):
	""" A 2D diffusion-Gaussian; no covariance """
	xG = 1.0/(np.sqrt(N*4*np.pi*t0))*np.repeat([np.exp(-(x*x)*0.25/t0)],N,axis=0)
	yG = 1.0/(np.sqrt(N*4*np.pi*(t0+t)))*np.repeat([np.exp(-(y*y)*0.25/(t0+t))],N,axis=0)
	pGD = np.dot(yG.T,xG)[0]
	return pGD
	
def Gauss2D(x,y,C):
	""" Covariant 2D Gaussian """
	X,Y = np.meshgrid(x,y)
	c = np.sqrt(C)
	return plt.mlab.bivariate_normal(X,Y, c[0,0],c[1,1], 0.0,0.0, c[0,1])

##================================================
##================================================
if __name__=="__main__":
	main()
