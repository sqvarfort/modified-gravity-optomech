import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import time
import os
from scipy import optimize
import yaml
from matplotlib import rcParams
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib as mpl

#############################

def S0line(M,hbar,c,Ms,rhobg,Rs):
	return ((3*Ms*np.sqrt(rhobg))/(Rs*8*np.pi*M**(3/2)))**(2/5)*hbar*c/e

def Seq(x, a, b):
	return x**2 + a*x**3 - b

def Sfuncold(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs):
	s2 = S2(M,Lambda,rhobg,Rs,Ms,hbar,c)
	if np.isscalar(s2):
		if s2 > 0:
			return Rs*np.sqrt(s2)
		else:
			return 0
	else:
		ret = np.zeros(s2.shape)
		nz = np.where(s2 > 0)
		ret[nz] = Rs*np.sqrt(s2[nz])
		return ret

def Sfunc(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs):
	"""
	Description: This function solves a cubic polynomial for S 
	Input: System parameters (M in units of Mp)
	Output: Scalar or array of S values
	"""
	# First compute the square of the rough estimate formula
	s2 = S2(M,Lambda,rhobg,Rs,Ms,hbar,c)

	# Check if a scalar or vector has been passed
	if np.isscalar(s2):
		# Check if the value of S^2 is positive or negative
		if s2 > 0:
			mbg = mbgfunc(M,Lambda,hbar,c,rhobg)
			phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
			phiS = np.sqrt(Lambda**5*M/(rhos*(hbar*c)**3))
			a = (2/3)*(1/(1 + mbg*Rs*c/hbar) - 1)
			b = 1 - 8*np.pi*M/(3*Ms*hbar*c)*Rs*(phibg - phiS) + (2/3)*(1/(1 + mbg*Rs*c/hbar)-1)

			# Find a zero in the interval 0 and 1
			xroot = optimize.brentq(Seq,0,1,args=(a,b))
			return xroot*Rs
		# If negative, return S = 0
		else:
			return 0
	# If instead we have an array
	else:
		# Define all quantities
		mbg = mbgfunc(M,Lambda,hbar,c,rhobg)
		phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
		phiS = np.sqrt(Lambda**5*M/(rhos*(hbar*c)**3))
		a = (2/3)*(1/(1 + mbg*Rs*c/hbar) - 1)
		b = 1 - 8*np.pi*M/(3*Ms*hbar*c)*Rs*(phibg - phiS) + (2/3)*(1/(1 + mbg*Rs*c/hbar)-1)
		
		# First find where S^2 is real and non-zero
		Svalues = np.zeros(s2.shape) 

		# Loop over the non-zero elements
		for i in range(0,s2.shape[0]):
			for j in range(0,s2.shape[0]):
				if s2[i,j] > 0:
					xroot = optimize.brentq(Seq,0,1,args=(a[i,j],b[i,j]))
					Svalues[i,j] = xroot*Rs
				else:
					Svalues[i,j] = 0
		return Svalues

# def S3Approxfunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,R):

def formA(x):
	return (1+x)*np.exp(-x)*(np.sinh(x)/x)

def formB(x):
	return (1+x)*np.exp(-x)*(np.cosh(x) - np.sinh(x)/x)

def phibgfunc(M,Lambda,rhobg,hbar,c):
	return np.sqrt(M*Lambda**5/(rhobg*(hbar*c)**3))

def S2(M,Lambda,rhobg,Rs,Ms,hbar,c):
	phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
	return 1 - 8*np.pi*(M/(3*Ms))*(Rs*phibg)/(hbar*c)


def xifuncold(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs):
	S = Sfunc(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs)
	s3 = 1 - S**3/Rs**3

	if np.isscalar(s3):
		if s3 < 1e-8:
			phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
			s3 = (4*np.pi*(M/(Ms))*(Rs*phibg)/(hbar*c))
		return s3
	else:
		phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
		small = np.where(s3 < 1e-8)
		s3[small] = (4*np.pi*(M[small]/(Ms))*(Rs*phibg[small])/(hbar*c))
		return s3

def xifunc(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs):
	"""
	Description: When S~Rs, return an approximate value for 1 - S**3/Rs**3 
	Input: System parameters
	Output: Array of s3 values
	"""
	S = Sfunc(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs)
	xi = 1 - S**3/Rs**3

	# Check if the cancellation is too small
	if np.isscalar(xi):
		if xi < 1e-8:
			# Compute quantities
			mbg = mbgfunc(M,Lambda,hbar,c,rhobg)
			phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
			phiS = np.sqrt(Lambda**5*M/(rhos*(hbar*c)**3))
			# Then compute xi
			xi = 4*np.pi*M/(Ms*hbar*c)*Rs*(phibg - phiS)*(1 + mbg*Rs*c/hbar) 
		return xi
	else:
		# Compute quantities
		mbg = mbgfunc(M,Lambda,hbar,c,rhobg)
		phibg = phibgfunc(M,Lambda,rhobg,hbar,c)
		phiS = np.sqrt(Lambda**5*M/(rhos*(hbar*c)**3))
		# Check for which values xi is small
		small = np.where(xi < 1e-8)
		# Replace those values with the approximation to avoid round-off errors
		xi[small] = 4*np.pi*M[small]/(Ms*hbar*c)*Rs*(phibg[small] - phiS[small])*(1 + mbg[small]*Rs*c/hbar) 
		return xi 

def mbgfunc(M,Lambda,hbar,c,rhobg):
	return (4*rhobg**3*(hbar*c)**9/(M**3*Lambda**5))**(1/4)/c**2

def kappafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr):
	# Convert to exponential
	M = 10**Mlog*Mp

	# Compute S and S3
	xiS = xifunc(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs)

	# Compute quantities
	mbg = mbgfunc(M,Lambda,hbar,c,rhobg)
	alpha = 2*(Mp**2/M**2)*xiS*(1/(1 + mbg*Rs*c/hbar))*np.exp(mbg*Rs*c/hbar)
	llambda = hbar/(mbg*c)

	# Compute and return kappa
	if probescr == True:
		xiP = xifunc(M,Lambda,Mp,hbar,c,Mprobe,rhos,rhop,rhobg,Rp)
		kappa = 2*(Mp/M)**2*xiP*xiS*np.exp(-c*mbg*r0/hbar)*(1 + c*mbg*r0/hbar)*(formA(c*mbg*Rp/hbar) + hbar*formB(c*mbg*Rp/hbar)*((c*mbg*r0 + 2*hbar)/(c*mbg*r0*(c*mbg*r0 + hbar))))*(1/(1 + mbg*Rs*c/hbar))*np.exp(mbg*Rs*c/hbar)
	if probescr == False:
		kappa = ((llambda + r0)/llambda)*alpha*np.exp(-r0/llambda)
	return kappa


def sigmafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr):
	# Convert to exponential
	M = 10**Mlog*Mp

	# Compute S and S3	
	xiS = xifunc(M,Lambda,Mp,hbar,c,Ms,rhos,rhop,rhobg,Rs)
	
	# Compute quantities
	mbg = mbgfunc(M,Lambda,hbar,c,rhobg)
	alpha = 2*(Mp**2/M**2)*xiS*(1/(1 + mbg*Rs*c/hbar))*np.exp(mbg*Rs*c/hbar)
	llambda = hbar/(mbg*c)

	if probescr == True:
		xiP = xifunc(M,Lambda,Mp,hbar,c,Mprobe,rhos,rhop,rhobg,Rp)
		sigma = 2*(Mp/M)**2*xiS*xiP*np.exp(-c*mbg*r0/hbar)*(formB(c*mbg*Rp/hbar)*((c**2*mbg**2*r0**2 + 4*c*mbg*r0*hbar + 6*hbar**2)/(c*mbg*r0*hbar)) + formA(c*mbg*Rp/hbar)*(c**2*mbg**2*r0**2/hbar**2 + 2*c*mbg*r0/hbar + 2))*(1/(1 + mbg*Rs*c/hbar))*np.exp(mbg*Rs*c/hbar)
	if probescr == False:
		sigma = (r0**2 + 2*r0*llambda + 2*llambda**2)/(llambda**2)*alpha*np.exp(- r0/llambda)
	return sigma

def Deltakappa(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Convert to exponential
	M = 10**Mlog*Mp
	gNewt = 1

	deltakappa = (1/np.sqrt(Mes))*(1./gNewt)*(1./(np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2.)))*(1./(8*np.pi*n*g0))*np.sqrt(2.*hbar*omegam**5/Mprobe)
	kappa = kappafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return kappa 


def Deltakappares(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Transform to base 10
	M = 10**Mlog*Mp
	gNewt = 1

	deltakappa = (1/np.sqrt(Mes))*(1/gNewt)*(1/np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)*1/(4*np.pi*n*g0)
	kappa = kappafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return kappa 

def Deltasigma(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Transform to base 10
	M = 10**Mlog*Mp
	gNewt = 1

	deltasigma =  (1/np.sqrt(Mes))*(1/gNewt)*(1/np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)*1/(4*np.pi*n*g0*epsilon)
	sigma = sigmafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return epsilon*sigma 

def Deltasigmares(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Define M*Mp
	M = 10**Mlog*Mp
	sigma = sigmafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return epsilon*sigma 


# Function to find all zeros, given a meshgrid:
def findAllZeros(func,X,Y,Z,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr,logX=True,logY=False):
    zeroList = []
    if logX:
        Xuse = np.log10(X)
    else:
        Xuse = X
    if logY:
        Yuse = np.log10(Y)
    else:
        Yuse = Y
    for k in range(0,len(X)):
        rowList = []
        for l in range(0,len(X[0])-1):
            if Z[k,l]*Z[k,l+1] < 0 and np.isfinite(Z[k,l]) and np.isfinite(Z[k, l+1]):
                # Found a zero:           
                xroot = optimize.brentq(func,Xuse[k,l],Xuse[k,l+1],args=(Yuse[k,l],Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr))
                yroot = Yuse[k,l]
                rowList.append((xroot,yroot))
        zeroList.append(rowList)
    return zeroList

def extractZerosLine(zeroList,line=1):
    zerox = np.zeros(len(zeroList))
    zeroy = np.zeros(len(zeroList))
    for k in range(0,len(zerox)):
        if len(zeroList[k]) > line - 1:
            zerox[k] = zeroList[k][line-1][0]
            zeroy[k] = zeroList[k][line-1][1]
    haveZeros = np.where(zerox != 0)[0]
    return [zerox[haveZeros],zeroy[haveZeros]]

########################################################
# Code start
########################################################
# Import arguments from config.yaml

config = 'config.yaml'

# Load arguments from yaml file 
args = {}
if type(config) == str:
    with open(config) as cfile:
        args.update(yaml.load(cfile))
elif type(config) == dict:
    args.update(config)
else:
    print("Failed to load config arguments")

hbar = float(args['hbar'])
G = float(args['G'])
c = float(args['c'])
e = float(args['e'])
muc = float(args['muc'])
g0 = float(args['g0'])
omegam = float(args['omegam'])
r0 = float(args['r0'])
epsilon = float(args['epsilon'])
r = float(args['r'])
n = float(args['n'])
Ms = float(args['Ms'])
Mes = float(args['Mes'])

Ms = float(args['Ms'])
rhos = float(args['rhos'])
rhop = float(args['rhop'])
Mprobe = float(args['Mprobe'])
rhobg = float(args['rhobg'])
Lambdamin = float(args['Lambdamin'])
Lambdamax = float(args['Lambdamax'])
Mmin = float(args['Mmin'])
Mmax = float(args['Mmax'])

nSample = int(args['nSample'])
#probescr = bool(args['probescr'])

# Compute some quantities
Mp = np.sqrt(hbar*c/(8*np.pi*G))
Rs = (3*Ms/(4*np.pi*rhos))**(1/3)
Rp = (3*Mprobe/(4*np.pi*rhop))**(1/3)

# Define the sample and the range of parameters
LambdaRange = e*10**(np.linspace(Lambdamin,Lambdamax,nSample))
MRange = 10**(np.linspace(Mmin,Mmax,nSample))

MGrid, LambdaGrid = np.meshgrid(MRange,LambdaRange)

# Compute without probe screening
probescr = False
DeltasigmaGridres = np.log10(Deltasigmares(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr))
probescr = True
DeltasigmaGridresScr = np.log10(Deltasigmares(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr))

# Plot the functions
rcParams.update({'figure.autolayout': True})
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['xtick.minor.width'] = 0
fig, ax = plt.subplots(figsize = (7, 6))
plt.xlabel('$M/M_{\\mathrm{P}}$', fontfamily = 'serif',  fontsize = 15)
plt.ylabel('$\\Lambda\,(\\mathrm{eV})$', fontfamily = 'serif',  fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

viridis = cm.get_cmap('viridis', 10)
hot = cm.get_cmap('plasma',10)
levels = [-15, -10, -5, -0, 5, 10, 15, 20, 25, 30]
CS = ax.contourf(MGrid, LambdaGrid/e, DeltasigmaGridres, levels = levels, colors = viridis.colors)
CSscr = ax.contour(MGrid, LambdaGrid/e, DeltasigmaGridresScr, levels = levels, colors = hot.colors)

plt.xscale('log')
plt.yscale('log')
ax.set_ylim(1e-12,1e2)
ax.set_xlim(1e-15,1e2)
clb = fig.colorbar(CS, extend = 'both')
clb.ax.tick_params(labelsize=12)
clb.add_lines(CSscr)

#clb.ax.plot(0.5, 0.5, 'w.') # my data is between 0 and 1
#clb.ax.plot(0.6, 0.6, 'b.') # my data is between 0 and 1
clb.set_label('$\\log_{10}(\\Delta F)$', labelpad=-40, y=1.05, rotation=0, fontsize = 12)
#ax.legend(loc = 'lower left', labelspacing = 0.4, fontsize = 12)
#ax.clabel(CS, inline=True, fontsize=10)
plt.savefig('MLambdaforceplot.pdf')
plt.show()
