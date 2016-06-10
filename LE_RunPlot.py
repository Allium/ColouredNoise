import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from platform import system
from datetime import datetime
from LE_Utils import plot_fontsizes

## ============================================================================

outdir = "Pressure/"+str(datetime.now().strftime("%y%m%d"))+"_CIR_DN_dt"
nosave = 1
annoloc = (0.02,0.87)
plt.rcParams.update({"axes.labelsize": plot_fontsizes()[0]})

## ============================================================================
## ============================================================================

def plot_step_wall(xy,rcoord,R,S,a,dt,vb):
	"""
	Distributions of spatial steps in wall regions
	"""
	me = "LE_RunPlot.plot_step_wall: "
	
	fig, axs = plt.subplots(2,2); axs = axs.reshape([axs.size])
	
	##-------------------------------------------------------------------------
	## distribution of x step
	ax = axs[0]
	xstep = np.hstack([np.diff(xy[:,0]),0])
	xstepin = xstep[rcoord<S]; xstepout = xstep[rcoord>R]
	hout, binout = ax.hist(xstepout,bins=50, label="R",color="b",alpha=0.5,normed=True)[0:2]
	hin, binin = ax.hist(xstepin,bins=binout,label="S",color="g",alpha=0.5,normed=True)[0:2]
	ax.set_xlim(right=-ax.get_xlim()[0])
	ax.grid();	ax.legend(loc="upper right")
	ax.set_xlabel("$\\delta x$"); ax.set_ylabel("$p(\\delta x)$")
	ax.set_title("x step")
	
	## Fit
	fitfunc = lambda x, A, b: A*np.exp(-0.5*b*(x*x))
	x = 0.5*(binin[:-1]+binin[1:])
	fitout = sp.optimize.curve_fit(fitfunc, x, hout)[0]
	fitin = sp.optimize.curve_fit(fitfunc, x, hin)[0]
	ax.annotate("scale: $R, S = "+str("%.2g")%(1/fitin[1])+", "+str("%.2g")%(1/fitout[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.plot(x,fitfunc(x,*fitin),"g-",lw=2);	ax.plot(x,fitfunc(x,*fitout),"b-",lw=2)
	
	## Prediction
	
	##-------------------------------------------------------------------------
	## distribution of y step
	ax = axs[1]
	ystep = np.hstack([np.diff(xy[:,1]),0])
	ystepin = ystep[rcoord<S]; ystepout = ystep[rcoord>R]
	hout, binout = ax.hist(xstepout,bins=50,label="R",color="b",alpha=0.5,normed=True)[0:2]
	hin, binin = ax.hist(xstepin,bins=binout,label="S",color="g",alpha=0.5,normed=True)[0:2]
	xl = min(np.abs(ax.get_xlim())); ax.set_xlim([-xl,xl])
	ax.grid()
	ax.set_xlabel("$\\delta y$"); ax.set_ylabel("$p(\\delta y)$")
	ax.set_title("y step")
	##
	fitfunc = lambda x, A, b: A*np.exp(-0.5*b*(x*x))
	x = 0.5*(binin[:-1]+binin[1:])
	fitout = sp.optimize.curve_fit(fitfunc, x, hout)[0]
	fitin = sp.optimize.curve_fit(fitfunc, x, hin)[0]
	ax.annotate("scale: $R, S = "+str("%.2g")%(1/fitin[1])+", "+str("%.2g")%(1/fitout[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.plot(x,fitfunc(x,*fitin),"g-",lw=2);	ax.plot(x,fitfunc(x,*fitout),"b-",lw=2)
	
	##-------------------------------------------------------------------------
	## distribution of r step
	ax = axs[2]
	rstep = np.hstack([np.diff(rcoord),0])
	rstepin = rstep[rcoord<S]; rstepout = rstep[rcoord>R]
	hout, binout = ax.hist(rstepout,bins=150,label="R",color="b",alpha=0.5,normed=True)[0:2]
	hin, binin = ax.hist(rstepin,bins=binout,label="S",color="g",alpha=0.5,normed=True)[0:2]
	xl = min(np.abs(ax.get_xlim())); ax.set_xlim([-xl,xl])
	ax.grid()
	ax.set_xlabel("$\\delta r$"); ax.set_ylabel("$p(\\delta r)$")
	ax.set_title("r step")
	##
	x = 0.5*(binin[:-1]+binin[1:])
	fitfunc = lambda x, A, b, mu: A*np.exp(-0.5*b*(x-mu)*(x-mu))
	fitout = sp.optimize.curve_fit(fitfunc, x, hout, p0=[100.0,100.0,0.0])[0]
	ax.plot(x,fitfunc(x,*fitout),"b-",lw=2)
	fitin  = sp.optimize.curve_fit(fitfunc, x, hin, p0=[100.0,100.0,0.0])[0]
	ax.plot(x,fitfunc(x,*fitin),"g-",lw=2)
	ax.annotate("scale: $R, S = "+str("%.2g")%(1/fitin[1])+", "+str("%.2g")%(1/fitout[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.annotate("peak: $R, S = "+str("%.2g")%(fitin[2])+", "+str("%.2g")%(fitout[2])+"$",
				xy=(0,0),xytext=(annoloc[0],annoloc[1]-0.05),textcoords="axes fraction")
				
	##-------------------------------------------------------------------------
	## distribution of phi
	ax = axs[3]
	dxy = np.diff(xy.T)
	phistep = np.hstack([np.arctan2(dxy[1],dxy[0]),0])
	phistepin = phistep[rcoord<S]; phistepout = phistep[rcoord>R]
	ang = np.linspace(-np.pi,np.pi,36)
	ax.hist(phistepin,bins=ang,label="S",color="b",alpha=0.5,normed=True)
	ax.hist(phistepout,bins=ang,label="R",color="g",alpha=0.5,normed=True)
	ax.plot(ang,1/(2*np.pi)*np.ones(ang.size),"k--",lw=2.0)
	ax.set_xlim(left=-np.pi,right=np.pi)
	ax.grid()
	ax.set_xlabel("$\\phi$"); ax.set_ylabel("$p(\\phi)$")
	ax.set_title("phi step")
	
	##-------------------------------------------------------------------------
	## Save
	fig.tight_layout()
	fig.suptitle("Spatial step statistics")
	plt.subplots_adjust(top=0.9)
	plotfile = outdir+str(dt)+\
				"/STEP_CIR_DN_a"+str(a)+"_R"+str(R)+"_S"+str(S)+"_dt"+str(dt)+".png"
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"STEP figure saved to",plotfile
	if vb:	plt.show()
	plt.close()
	
	return
	
## ============================================================================
	
def plot_eta_wall(xy,rcoord,exy,ercoord,R,S,a,dt,vb):
	"""
	Plot eta in wall regions
	"""
	me = "LE_RunPlot.plot_eta_stats: "
	
	fig, axs = plt.subplots(2,2); axs = axs.reshape([axs.size])
	
	##-------------------------------------------------------------------------
	## distribution of etax
	ax = axs[0]
	
	xeta = exy[:,0]
	xetain = xeta[rcoord<S]; xetaout = xeta[rcoord>R]
	hout, binout = ax.hist(xetaout,bins=50,label="R",color="b",alpha=0.5,normed=True)[0:2]
	hin, binin = ax.hist(xetain,bins=binout,label="S",color="g",alpha=0.5,normed=True)[0:2]
	
	xl = min(np.abs(ax.get_xlim())); ax.set_xlim([-xl,xl])
	ax.grid();	ax.legend(loc="upper right")
	ax.set_xlabel("$\\eta_x$"); ax.set_ylabel("$p(\\eta_x)$")
	ax.set_title("eta x")

	## Fit
	x = 0.5*(binin[:-1]+binin[1:])
	fitfunc = lambda x, A, b, mu: A*np.exp(-0.5*b*(x-mu)*(x-mu))
	fitout = sp.optimize.curve_fit(fitfunc, x, hout)[0]
	fitin = sp.optimize.curve_fit(fitfunc, x, hin)[0]
	ax.plot(x,fitfunc(x,*fitin),"g-",lw=2);	ax.plot(x,fitfunc(x,*fitout),"b-",lw=2)
	ax.annotate("scale: $R, S = "+str("%.2g")%(1/fitin[1])+", "+str("%.2g")%(1/fitout[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.annotate("peak: $R, S = "+str("%.2g")%(fitin[2])+", "+str("%.2g")%(fitout[2])+"$",
				xy=(0,0),xytext=(annoloc[0],annoloc[1]-0.05),textcoords="axes fraction")
				
	##-------------------------------------------------------------------------
	## distribution of etay
	ax = axs[1]
	
	yeta = exy[:,1]
	yetain = yeta[rcoord<S]; yetaout = yeta[rcoord>R]
	hout, binout = ax.hist(yetaout,bins=50,label="R",color="b",alpha=0.5,normed=True)[0:2]
	hin, binin = ax.hist(yetain,bins=binout,label="S",color="g",alpha=0.5,normed=True)[0:2]
	
	xl = min(np.abs(ax.get_xlim())); ax.set_xlim([-xl,xl])
	ax.grid()
	ax.set_title("eta y")
	ax.set_xlabel("$\\eta_x$"); ax.set_ylabel("$p(\\eta_x)$")

	## Fit
	x = 0.5*(binin[:-1]+binin[1:])
	fitfunc = lambda x, A, b, mu: A*np.exp(-0.5*b*(x-mu)*(x-mu))
	fitout = sp.optimize.curve_fit(fitfunc, x, hout)[0]
	fitin = sp.optimize.curve_fit(fitfunc, x, hin)[0]
	ax.plot(x,fitfunc(x,*fitin),"g-",lw=2);	ax.plot(x,fitfunc(x,*fitout),"b-",lw=2)
	ax.annotate("scale: $R, S = "+str("%.2g")%(1/fitin[1])+", "+str("%.2g")%(1/fitout[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.annotate("peak: $R, S = "+str("%.2g")%(fitin[2])+", "+str("%.2g")%(fitout[2])+"$",
				xy=(0,0),xytext=(annoloc[0],annoloc[1]-0.05),textcoords="axes fraction")
			
	##-------------------------------------------------------------------------
	## distribution of er
	ax = axs[2]
	
	erin = ercoord[rcoord<S]; erout = ercoord[rcoord>R]
	hout, binout = ax.hist(erout,bins=150,label="R",color="b",alpha=0.5,normed=True)[0:2]
	hin, binin = ax.hist(erin,bins=binout,label="S",color="g",alpha=0.5,normed=True)[0:2]
	
	ax.set_xlim(left=0,right=round(binout[-1],0))
	ax.grid()
	ax.set_xlabel("$\\eta_r$"); ax.set_ylabel("$p(\\eta_r)$")
	ax.set_title("eta r")
	
	## Fit
	x = 0.5*(binin[:-1]+binin[1:])
	fitfunc = lambda x, A, b: A*x*np.exp(-0.5*b*(x*x))
	fitout = sp.optimize.curve_fit(fitfunc, x, hout, p0=[100.0,1.0])[0]
	ax.plot(x,fitfunc(x,*fitout),"b-",lw=2)
	fitin  = sp.optimize.curve_fit(fitfunc, x, hin, p0=[100.0,1.0])[0]
	ax.plot(x,fitfunc(x,*fitin),"g-",lw=2)
	ax.annotate("scale: $R, S = "+str("%.2g")%(1/fitin[1])+", "+str("%.2g")%(1/fitout[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
				
	##-------------------------------------------------------------------------
	## distribution of ephi
	ax = axs[3]
	
	ang = np.linspace(-np.pi,np.pi,36)
	epcoord = np.arctan2(exy[:,1],exy[:,0])
	epin = epcoord[rcoord<S]; epout = epcoord[rcoord>R]
	
	ax.hist(epin,bins=ang,label="S",color="b",alpha=0.5,normed=True)
	ax.hist(epout,bins=ang,label="R",color="g",alpha=0.5,normed=True)
	ax.plot(ang,1/(2*np.pi)*np.ones(ang.size),"k--",lw=2.0)
	
	ax.set_xlim(left=-np.pi,right=np.pi)
	ax.grid()
	ax.set_title("eta phi")
	ax.set_xlabel("$\\eta_\\phi$"); ax.set_ylabel("$p(\\eta_\\phi)$")
				

	##-------------------------------------------------------------------------
	## Save
	fig.tight_layout()
	fig.suptitle("eta statistics")
	plt.subplots_adjust(top=0.9)
	plotfile = outdir+str(dt)+\
				"/ETAS_CIR_DN_a"+str(a)+"_R"+str(R)+"_S"+str(S)+"_dt"+str(dt)+".png"
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"ETAS figure saved to",plotfile
	if vb:	plt.show()
	plt.close()
	
	return
	
## ============================================================================
## ============================================================================

def plot_step_bulk(xy,rcoord,ercoord,R,S,a,dt,vb):
	"""
	Distributions of spatial steps in wall regions
	"""
	me = "LE_RunPlot.plot_step_bulk: "
	
	fig, axs = plt.subplots(2,2); axs = axs.reshape([axs.size])
	
	##-------------------------------------------------------------------------
	## distribution of r step
	ax = axs[0]
	rstep = np.hstack([np.diff(rcoord),0])
	rstepb = rstep[(rcoord>S)*(rcoord<R)]
	ha, bina = ax.hist(rstep,bins=100,label="A",color="b",alpha=0.5,normed=True)[0:2]
	hb, binb = ax.hist(rstepb,bins=bina,label="B",color="g",alpha=0.5,normed=True)[0:2]
	ax.grid()
	xl = min(np.abs(ax.get_xlim())); ax.set_xlim([-xl,xl])
	ax.set_title("r step")
	ax.set_xlabel("$\\delta r$"); ax.set_ylabel("$p(\\delta r)$")
	
	## Fit
	x = 0.5*(binb[:-1]+binb[1:])
	fitfunc = lambda x, A, b, mu: A*np.exp(-0.5*b*(x-mu)*(x-mu))
	fitb = sp.optimize.curve_fit(fitfunc, x, hb, p0=[np.sqrt(a/(2*np.pi*dt*dt)),a/(dt*dt),0.0])[0]
	ax.plot(x,fitfunc(x,*fitb),"g-",lw=2)
	ax.annotate("scale: $B = "+str("%.2g")%(1/fitb[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.annotate("peak: $B = "+str("%.2g")%(fitb[2])+"$",
				xy=(0,0),xytext=(annoloc[0],annoloc[1]-0.05),textcoords="axes fraction")
				
	## Prediction
	ax.plot(x,fitfunc(x,np.sqrt(a/(2*np.pi*dt*dt)),a/(dt*dt),0.0), "k--", lw=2.0)
				
	##-------------------------------------------------------------------------
	## distribution of er step
	ax = axs[1]
	erstep = np.hstack([np.diff(ercoord),0])
	erstepb = erstep[(rcoord>S)*(rcoord<R)]
	hb, binb = ax.hist(erstepb,bins=100,label="B",color="g",alpha=0.5,normed=True)[0:2]
	ha, bina = ax.hist(erstep,bins=binb,label="A",color="b",alpha=0.5,normed=True)[0:2]
	ax.grid()
	xl = min(np.abs(ax.get_xlim())); ax.set_xlim([-xl,xl])
	ax.set_title("eta r step")
	ax.set_xlabel("$\\delta\\eta_r$"); ax.set_ylabel("$p(\\delta\\eta_r)$")
	
	## Fit
	x = 0.5*(binb[:-1]+binb[1:])
	fitfunc = lambda x, A, b, mu: A*np.exp(-0.5*b*(x-mu)*(x-mu))
	fitb = sp.optimize.curve_fit(fitfunc, x, hb, p0=[100.0,1.0,0.0])[0]
	ax.plot(x,fitfunc(x,*fitb),"g-",lw=2)
	ax.annotate("scale: $B = "+str("%.2g")%(1/fitb[1])+"$",
				xy=(0,0),xytext=annoloc,textcoords="axes fraction")
	ax.annotate("peak: $B = "+str("%.2g")%(fitb[2])+"$",
				xy=(0,0),xytext=(annoloc[0],annoloc[1]-0.05),textcoords="axes fraction")
				
	## Prediction
	# ax.plot(x, fitfunc(x,))
	
	##-------------------------------------------------------------------------
	## distribution of phi -- heading
	ax = axs[2]
	dxy = np.diff(xy.T)
	phistepa = np.hstack([np.arctan2(dxy[1],dxy[0]),0])
	phistepb = phistepa[(S<rcoord)*(rcoord<R)]
	ang = np.linspace(-np.pi,np.pi,36)
	ax.hist(phistepa,bins=ang,label="A",color="b",alpha=0.5,normed=True)
	ax.hist(phistepb,bins=ang,label="B",color="g",alpha=0.5,normed=True)
	ax.plot(ang,1/(2*np.pi)*np.ones(ang.size),"k--",lw=2.0)
	ax.set_xlim(left=-np.pi,right=np.pi)
	ax.grid()
	ax.set_xlabel("$\\phi$"); ax.set_ylabel("$p(\\phi)$")
	ax.set_title("phi distribution")
	
	##-------------------------------------------------------------------------
	## Correlation function of phi
	ax = axs[3]
	ftphia = np.fft.rfft(phistepa)
	corra = np.fft.irfft(ftphia*np.conj(ftphia))
	ftphib = np.fft.rfft(phistepb)
	corrb = np.fft.irfft(ftphib*np.conj(ftphib))
	t = np.arange(0.0,10.0*a/2.0,dt)
	ax.plot(t,corra[:t.size]/corra[0], label="All")
	ax.plot(t,corrb[:t.size]/corrb[0], label="Bulk")
	ax.grid(); ax.legend()
	ax.set_xlabel("$t$"); ax.set_ylabel("$\\langle\\phi(t)\\phi(t-t^\\prime)\\rangle_t$")
	ax.set_title("phi correlation")

	## Fit 
	ax.plot(t,np.exp(-2.0*t/a),"k--",lw=2.0)
	
	##-------------------------------------------------------------------------
	## Save
	fig.tight_layout()
	fig.suptitle("Bulk spatial step statistics")
	plt.subplots_adjust(top=0.9)
	plotfile = outdir+str(dt)+\
				"/BULK_CIR_DN_a"+str(a)+"_R"+str(R)+"_S"+str(S)+"_dt"+str(dt)+".png"
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"BULK figure saved to",plotfile
	if vb:	plt.show()
	plt.close()
	
	return
	
## ============================================================================
	
def plot_eta_bulk(xy,rcoord,exy,ercoord,R,S,a,dt,vb):	
	return
	
## ============================================================================
## ============================================================================

def plot_traj(xy,rcoord,R,S,lam,nu,force_dnu,a,dt,vb):
	"""
	"""
	me = "LE_RunPlot.plot_traj: "

	fig = plt.figure(); ax = fig.gca()
	cm = plt.get_cmap("winter"); CH = 100; NPOINTS = xy.shape[0]/CH
	if system() == "Linux":
		ax.set_prop_cycle("color",[cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
	else:
		ax.set_color_cycle([cm(1.*i/(NPOINTS-1)) for i in range(NPOINTS-1)])
	
	for i in range(NPOINTS-1):
		ax.plot(xy[i*CH:(i+1)*CH+1,0],xy[i*CH:(i+1)*CH+1,1],zorder=1)
		
	## Plot force arrows
	if 1:
		numarrow = 100	## not number of arrows plotted
		for j in range(0, xy.shape[0], xy.shape[0]/numarrow):
			uforce = force_dnu(xy[j],rcoord[j],R,S,lam,nu)
			if uforce.any()>0.0:
				plt.quiver(xy[j,0],xy[j,1], uforce[0],uforce[1],
						width=0.008, scale=20.0, color="purple", headwidth=2, headlength=2, zorder=2)
		print me+"Warning: angles look off."
							
	## Plot walls
	ang = np.linspace(0.0,2*np.pi,360)
	ax.plot(R*np.cos(ang),R*np.sin(ang),"y--",(R+lam)*np.cos(ang),(R+lam)*np.sin(ang),"y-",lw=2.0, zorder=3)
	ax.plot(S*np.cos(ang),S*np.sin(ang),"r--",(S-lam)*np.cos(ang),(S-lam)*np.sin(ang),"r-",lw=2.0, zorder=3)
	
	ax.set_xlim((-R-lam-0.1,R+lam+0.1));	ax.set_ylim((-R-lam-0.1,R+lam+0.1))
	ax.grid()
	
	## Save
	plotfile = outdir+str(dt)+\
				"/TRAJ_CIR_DN_a"+str(a)+"_R"+str(R)+"_S"+str(S)+\
				"_l"+str(lam)+"_n"+str(nu)+"_t"+str(round(rcoord.size*dt/5e2,1))+"_dt"+str(dt)+".png"
	if not nosave:
		fig.savefig(plotfile)
		if vb:	print me+"TRAJ figure saved to",plotfile
	if vb:	plt.show()
	plt.close()
	
	return
	
## ============================================================================
