import fipy as fp
import numpy as np
from matplotlib import pyplot as plt

nx = 20
ny = nx
dx = 1.
dy = dx
L = dx * nx
mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)

phi = fp.CellVariable(name = "solution variable",mesh = mesh,value = 0.)

D = 1.
eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=D)

valueTopLeft = 0
valueBottomRight = 1

X, Y = mesh.faceCenters
facesTopLeft = ((mesh.facesLeft & (Y > L / 2))| (mesh.facesTop & (X < L / 2)))
facesBottomRight = ((mesh.facesRight & (Y < L / 2))| (mesh.facesBottom & (X > L / 2)))

phi.constrain(valueTopLeft, facesTopLeft)
phi.constrain(valueBottomRight, facesBottomRight)

timeStepDuration = 10 * 0.9 * dx**2 / (2 * D)
steps = 10
for step in range(steps):
	eq.solve(var=phi,dt=timeStepDuration)

plt.contourf(phi.reshape([nx,ny]), extent=(-1,1,-1,1))
plt.show()
