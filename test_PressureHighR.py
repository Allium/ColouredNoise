import numpy as np
import scipy.optimize
from sys import argv
from scipy.special import erf, gamma
from matplotlib import pyplot as plt

from LE_Utils import plot_fontsizes
fsa,fsl,fst = plot_fontsizes()

def main():
	"""
	Plot P(a) for different potentials at a fixed (large) R.
	(Large R means we can ignore the difference between Pin and Pout -- in theory!)
	Compare with white noise solution, or with exact solution for zero-bulk quadratic potential.
	For calculations see notes 19/06/2016.
	For data see files 19/06/2016.
	"""
	
	try:
		R = float(argv[1])
		assert R in [50.0,100.0]
	except:
		R = 100.0
	DR = [0.0,1.0,10.0]
	lam = 1.0
	a=np.array([0.05,0.1,0.2,1.0,2.0,3.0,5.0,8.0,10.0])
	
	## RAW pressures -- will be normalised by WN prediction in next step
	
	if R==50.0:
		tit = "R=50"
		## Quadratic
		PR_q0b = np.array([0.001321,0.001249,0.001190,0.000902,0.000765,0.000661,0.000547,0.000451,0.000412])
		PL_q0b = np.array([0.001282,0.001230,0.001198,0.000900,0.000722,0.000614,0.000492,0.000396,0.000355])
		PR_q1b = np.array([0.000956,0.000914,0.000879,0.000717,0.000603,0.000537,0.000450,0.000382,0.000352])
		PL_q1b = np.array([0.000952,0.000908,0.000895,0.000673,0.000568,0.000489,0.000403,0.000333,0.000299])
		PR_q10b= np.array([0.000312,0.000299,0.000278,0.000252,0.000245,0.000237,0.000220,0.000209,0.000203])
		PL_q10b= np.array([0.000282,0.000290,0.000282,0.000244,0.000243,0.000222,0.000199,0.000172,0.000161])
		## Log-quadratic
		PR_lq0b = np.array([0.002126,0.001918,0.001750,0.001100,0.000838,0.000710,0.000571,0.000468,0.000421])
		PL_lq0b = np.array([0.002080,0.001913,0.001722,0.001032,0.000792,0.000652,0.000514,0.000405,0.000365])
		PR_lq1b = np.array([0.001376,0.001256,0.001141,0.000792,0.000635,0.000553,0.000467,0.000389,0.000355])
		PL_lq1b = np.array([0.001375,0.001235,0.001138,0.000764,0.000602,0.000513,0.000413,0.000334,0.000305])
		PR_lq10b= np.array([0.000327,0.000314,0.000290,0.000287,0.000251,0.000254,0.000224,0.000208,0.000198])
		PL_lq10b= np.array([0.000339,0.000325,0.000292,0.000261,0.000239,0.000218,0.000195,0.000176,0.000165])
	elif R==100.0: 
		tit = "R=100"
		## Quadratic
		PR_q0b = np.array([0.000653,0.000623,0.000591,0.000453,0.000375,0.000324,0.000266,0.000218,0.000199])
		PL_q0b = np.array([0.000651,0.000619,0.000583,0.000449,0.000357,0.000314,0.000255,0.000205,0.000185])
		PR_q1b = np.array([0.000479,0.000450,0.000436,0.000348,0.000295,0.000261,0.000221,0.000185,0.000168])
		PL_q1b = np.array([0.000478,0.000462,0.000441,0.000340,0.000288,0.000248,0.000208,0.000170,0.000156])
		PR_q10b= np.array([0.000141,0.000142,0.000127,0.000123,0.000112,0.000109,0.000104,0.000094,0.000091])
		PL_q10b= np.array([0.000146,0.000137,0.000138,0.000120,0.000116,0.000105,0.000098,0.000086,0.000082])
		## Log-quadratic
		PR_lq0b = np.array([0.001060,0.000957,0.000865,0.000534,0.000410,0.000348,0.000278,0.000227,0.000205])
		PL_lq0b = np.array([0.001043,0.000953,0.000870,0.000522,0.000400,0.000333,0.000262,0.000209,0.000187])
		PR_lq1b = np.array([0.000679,0.000620,0.000573,0.000395,0.000315,0.000269,0.000226,0.000186,0.000172])
		PL_lq1b = np.array([0.000687,0.000623,0.000561,0.000376,0.000301,0.000259,0.000210,0.000173,0.000156])
		PR_lq10b= np.array([0.000156,0.000149,0.000147,0.000122,0.000115,0.000115,0.000105,0.000095,0.000091])
		PL_lq10b= np.array([0.000161,0.000148,0.000139,0.000129,0.000117,0.000104,0.000097,0.000087,0.000082])
	
	## Normalise by WN calculated value
	PR_q0b /= PWN_q(R,R)
	PL_q0b /= PWN_q(R,R)
	PR_q1b /= PWN_q(R,R-1.0)
	PL_q1b /= PWN_q(R,R-1.0)
	PR_q10b/= PWN_q(R,R-10.0)
	PL_q10b/= PWN_q(R,R-10.0)
	PR_lq0b /= PWN_lq(R,R,lam)
	PL_lq0b /= PWN_lq(R,R,lam)
	PR_lq1b /= PWN_lq(R,R-1.0,lam)
	PL_lq1b /= PWN_lq(R,R-1.0,lam)
	PR_lq10b/= PWN_lq(R,R-10.0,lam)
	PL_lq10b/= PWN_lq(R,R-10.0,lam)

	## FItting
	fitfunc = lambda x, A, m:	A*np.power(x,m)
	# print "PR_q0b exponent fit",scipy.optimize.curve_fit(fitfunc, a[2:]+1, PR_q0b[2:], p0=[1.0,-0.5])[0][1]
	# print "PR_q1b exponent fit",scipy.optimize.curve_fit(fitfunc, a[2:]+1, PR_q1b[2:], p0=[1.0,-0.5])[0][1]
	# print "PR_q10b exponent fit",scipy.optimize.curve_fit(fitfunc, a[2:]+1, PR_q10b[2:], p0=[1.0,-0.5])[0][1]
	# print "PR_lq0b exponent fit",scipy.optimize.curve_fit(fitfunc, a[4:]+1, PR_lq0b[4:], p0=[1.0,-0.5])[0][1]
	# print "PR_lq1b exponent fit",scipy.optimize.curve_fit(fitfunc, a[4:]+1, PR_lq1b[4:], p0=[1.0,-0.5])[0][1]
	# print "PR_lq10b exponent fit",scipy.optimize.curve_fit(fitfunc, a[4:]+1, PR_lq10b[4:], p0=[1.0,-0.5])[0][1]
	
	## Plotting
	fig = plt.figure(); ax = fig.gca()
	
	aa = np.linspace(0.0,a[-1],100)
	ax.plot(aa+1,(aa+1)**-0.5,"b:",lw=2,label="$(\\alpha+1)^{-1/2}$")
	# ax.plot(aa+1,(aa+1)**-0.25,"k:",lw=2,label="$(\\alpha+1)^{-1/4}$")
		
	if 0:
		## Plot for both potentials
		tit = "$P_{\\rm out}$ for both potentials; "+tit
		ax.plot(a+1,PR_q0b,"o-", label="Quadratic, $R-S=0$")
		# ax.errorbar(a+1, PR_q0b, yerr=0.04*PR_q0b, ecolor='g', capthick=1)
		ax.plot(a+1,PR_q1b,"o-", label="Quadratic, $R-S=1$")
		ax.plot(a+1,PR_q10b,"o-", label="Quadratic, $R-S=10$")
		ax.set_color_cycle(None)
		ax.plot(a+1,PR_lq0b,"v--", label="Log-quadratic, $R-S=0$")
		ax.plot(a+1,PR_lq1b,"v--", label="Log-quadratic, $R-S=1$")
		ax.plot(a+1,PR_lq10b,"v--", label="Log-quadratic, $R-S=10$")
	elif 1:
		## Plot Pout, Pin for quadratic
		tit = "$P_{\\rm out}$ and $P_{\\rm in}$ for quadratic potential; "+tit
		ax.plot(a+1,PR_q0b,"o-", label="$P_{\\rm out}$, $R-S=0$")
		ax.plot(a+1,PL_q0b,"v--", c=ax.lines[-1].get_color(), label="$P_{\\rm in}$, $R-S=0$")
		ax.plot(a+1,PR_q1b,"o-", label="$P_{\\rm out}$, $R-S=1$")
		ax.plot(a+1,PL_q1b,"v--", c=ax.lines[-1].get_color(), label="$P_{\\rm in}$, $R-S=1$")
		ax.plot(a+1,PR_q10b,"o-", label="$P_{\\rm out}$, $R-S=10$")
		ax.plot(a+1,PL_q10b,"v--", c=ax.lines[-1].get_color(), label="$P_{\\rm in}$, $R-S=10$")
	elif 1:
		## Plot Pout, Pin for quadratic but against R
		tit = "$P_{\\rm out}$ and $P_{\\rm in}$ for quadratic potential; "+tit
		datR = np.array([PR_q0b,PR_q1b,PR_q10b]).T
		datL = np.array([PL_q0b,PL_q1b,PL_q10b]).T
		for i in range(0,a.size,2):
			ax.plot(DR,datR[i],"o-", label="$P_{\\rm out}$, $R-S=0$")
			ax.plot(DR,datL[i],"v--", c=ax.lines[-1].get_color(), label="$P_{\\rm out}$, $R-S=0$")
	

	# ax.set_yscale("log")
	# ax.set_xscale("log")
	# ax.set_xlim([1.0,12.0])
	
	ax.set_xlabel("$\\alpha+1$",fontsize=fsa)
	ax.set_ylabel("Pressure (normalised)",fontsize=fsa)
	ax.legend(fontsize=12)
	ax.grid(which="minor")
	fig.suptitle(tit)
	
	# fig.savefig("./Pressure/PQa_R"+str(R)+"_loglog.jpg")
	plt.show()
	
	return

## ============================================================================
	
def PWN_q(R,S):
	"""
	White noise pressure for quadratic potential.
	Assume R, S are large.
	"""
	assert R>=S
	# return 1/(2*np.pi)*1/(np.exp(-0.5*S*S) + 0.5*(R*R-S*S) + np.sqrt(np.pi/2)*(S*erf(S/np.sqrt(2))+R))
	return 1/(2*np.pi)*1/(0.5*(R*R-S*S)+np.sqrt(np.pi/2)*(R+S))
	
def PWN_lq(R,S,l):
	"""
	White noise pressure for log-quadratic potential.
	"""
	assert R>=S
	return 1/(2*np.pi)*1/(np.sqrt(np.pi)/2*l*(R+S)*gamma(l+1)/gamma(l+1.5) + 0.5*(R*R-S*S))
					
def PCN_q0b(R,a):
	"""
	Coloured noise pressure for quadratic potential with zero bulk
	Assume R large.
	"""
	# P = 1/(a+1) * 1/(2*np.pi)*1/(1/(a+1)*np.exp(-0.5*(a+1)*R*R) +\
					# np.sqrt(np.pi/(2*(a+1)))*R*(erf(R*np.sqrt((a+1)/2))+1.0))
	return 1/(2*np.pi)**1.5 * 1/(R*np.sqrt(a+1))
					
					
## ============================================================================
if __name__=="__main__":
	main()