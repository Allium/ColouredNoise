import fipy as fp
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

def main():
	"""
	NAME		2DFPE.py
	
	PURPOSE		Integrate time-dependent FPE
	
	EXECUTION	python 2DFPE.py
	
	STARTED		CS 24/09/2015
	"""
	
	nx = 20
	ny = nx
	dx = 1.
	dy = dx
	L = dx * nx
	mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
	y = np.linspace(-ny*dy*0.5,+ny*dy*0.5,ny,endpoint=True)[:,np.newaxis]
	X, Y = mesh.faceCenters

	## phi is pdf; psi is y*pdf for convenience
	phi = fp.CellVariable(mesh = mesh, value = 1/(dx*nx*dy*ny))
	psi = fp.CellVariable(mesh = mesh, value = (y*phi.value.reshape((nx,ny))).flatten())

	diffCoeff = 1.; diffMatri = [[0.,0.],[0.,diffCoeff]]
	convCoeff = np.array([-1.,+1.])
	eq = fp.TransientTerm(var=phi) == fp.DiffusionTerm(coeff=[diffMatri],var=phi) + 0*fp.ExponentialConvectionTerm(coeff=convCoeff,var=psi) 

	##---------------------------------------------------------------------------------------------------------
	## BCs
	phi = BC_value_at_boundary(phi,mesh)

	## Evolution
	timeStepDuration = 0.5*min(dy**2/(2.*diffCoeff),dy/(nx*dx))
	steps = 10
	for step in range(steps):
		print np.trapz(np.trapz(phi.value.reshape([nx,ny]),dx=dx),dx=dy)
		psi.value = (y*phi.value.reshape((nx,ny))).flatten()
		eq.solve(var=phi,dt=timeStepDuration)
		print phi.value[5]
		
	# plot_pdf(phi.value.reshape([nx,ny]),step+1)
	plt.contourf(phi.value.reshape([nx,ny]), extent=(-1,1,-1,1))
	plt.colorbar()
	plt.title("Density at timestep "+str(steps))
	plt.xlabel("$x$",fontsize=18); plt.ylabel("$\eta$",fontsize=18)
	plt.savefig("fig_FPE/Imp"+str(steps)+".png")
	plt.show()
	
	return

##=============================================================================================================
	
def BC_value_at_boundary(phi,mesh,value=(0.,0.)):	
	valueLeft, valueRight = value
	phi.constrain(valueLeft, mesh.facesLeft)
	phi.constrain(valueRight, mesh.facesRight)
	return phi
def BC_something(phi,mesh,value=(0.,0.)):
	valueTopLeft = 0
	valueBottomRight = 0
	facesTopLeft = (mesh.facesLeft & (Y > L / 2))
	facesBottomRight = (mesh.facesRight & (Y < L / 2))
	# Free flow elsewhere
	phi.constrain(valueTopLeft, facesTopLeft)
	phi.constrain(valueBottomRight, facesBottomRight)
	return phi

def plot_pdf(f,l):
	plt.contourf(f, extent=(-1,1,-1,1))
	plt.colorbar()
	plt.title("Density at timestep "+str(l))
	plt.xlabel("$x$",fontsize=18); plt.ylabel("$\eta$",fontsize=18)
	plt.savefig("fig_FPE/Imp"+str(l)+".png")
	plt.show()
	return
	
##=============================================================================================================
if __name__=="__main__":
	main()