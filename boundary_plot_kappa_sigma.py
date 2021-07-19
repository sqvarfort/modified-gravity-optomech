import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
import time
import os
from scipy import optimize
import yaml
from matplotlib import rcParams

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
	gNewt = G*Ms/r0**2

	deltakappa = (1/np.sqrt(Mes))*(1./gNewt)*(1./(np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2.)))*(1./(8*np.pi*n*g0))*np.sqrt(2.*hbar*omegam**5/Mprobe)
	kappa = kappafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return deltakappa/kappa - 1


def Deltakappares(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Transform to base 10
	M = 10**Mlog*Mp
	gNewt = G*Ms/r0**2

	deltakappa = (1/np.sqrt(Mes))*(1/gNewt)*(1/np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)*1/(4*np.pi*n*g0)
	kappa = kappafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return deltakappa/kappa - 1

def Deltasigma(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Transform to base 10
	M = 10**Mlog*Mp
	gNewt = G*Ms/r0**2

	deltasigma =  (1/np.sqrt(Mes))*(1/gNewt)*(1/np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)*1/(4*np.pi*n*g0*epsilon)
	sigma = sigmafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return deltasigma/sigma - 1

def Deltasigmares(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr):
	# Define M*Mp
	M = 10**Mlog*Mp
	gNewt = G*Ms/r0**2

	deltasigma = (1/np.sqrt(Mes))*(1/gNewt)*(1/np.sqrt(np.abs(muc)**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)*1/(2*np.pi**2*n**2*g0*epsilon)
	sigma = sigmafunc(Mlog,Lambda,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,r0,Rs,Rp,probescr)
	return deltasigma/sigma - 1


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

#Lambda = e*1e-4
#M = Mp*1e-15

# Define the sample and the range of parameters
LambdaRange = e*10**(np.linspace(Lambdamin,Lambdamax,nSample))
MRange = 10**(np.linspace(Mmin,Mmax,nSample))

# Compute the S for the source
Ssourceline = np.zeros(LambdaRange.shape)
Sprobeline = np.zeros(LambdaRange.shape)

for i in range(0, MRange.size):
	Ssourceline[i] = S0line(MRange[i]*Mp,hbar,c,Ms,rhobg,Rs)
	Sprobeline[i] = S0line(MRange[i]*Mp,hbar,c,Mprobe,rhobg,Rp)


MGrid, LambdaGrid = np.meshgrid(MRange,LambdaRange)

# Compute without probe screening
probescr = False
DeltakappaGrid = Deltakappa(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
DeltakappaGridres = Deltakappares(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
DeltasigmaGrid = Deltasigma(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
DeltasigmaGridres = Deltasigmares(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)

# Zeroes of Delta kappa 
zeroListDeltakappa = findAllZeros(Deltakappa,MGrid,LambdaGrid,DeltakappaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1kappa,zeroy1kappa] = extractZerosLine(zeroListDeltakappa,line=1)
[zerox2kappa,zeroy2kappa] = extractZerosLine(zeroListDeltakappa,line=2)
zeroListDeltakappares = findAllZeros(Deltakappares,MGrid,LambdaGrid,DeltakappaGridres,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1kappares,zeroy1kappares] = extractZerosLine(zeroListDeltakappares,line=1)
[zerox2kappares,zeroy2kappares] = extractZerosLine(zeroListDeltakappares,line=2)

# Zeros of Deltasigma 
zeroListDeltasigma = findAllZeros(Deltasigma,MGrid,LambdaGrid,DeltasigmaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1sigma,zeroy1sigma] = extractZerosLine(zeroListDeltasigma,line=1)
[zerox2sigma,zeroy2sigma] = extractZerosLine(zeroListDeltasigma,line=2)
zeroListDeltasigmares = findAllZeros(Deltasigmares,MGrid,LambdaGrid,DeltasigmaGridres,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1sigmares,zeroy1sigmares] = extractZerosLine(zeroListDeltasigmares,line=1)
[zerox2sigmares,zeroy2sigmares] = extractZerosLine(zeroListDeltasigmares,line=2)

# Compute with probe screening
probescr = True
DeltakappaGridscr = Deltakappa(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
DeltakappaGridresscr = Deltakappares(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
DeltasigmaGridscr = Deltasigma(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
DeltasigmaGridresscr = Deltasigmares(np.log10(MGrid),LambdaGrid,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)

zeroListDeltakappascr = findAllZeros(Deltakappa,MGrid,LambdaGrid,DeltakappaGridscr,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1kappascr,zeroy1kappascr] = extractZerosLine(zeroListDeltakappascr,line=1)
[zerox2kappascr,zeroy2kappascr] = extractZerosLine(zeroListDeltakappascr,line=2)
zeroListDeltakapparesscr = findAllZeros(Deltakappares,MGrid,LambdaGrid,DeltakappaGridresscr,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1kapparesscr,zeroy1kapparesscr] = extractZerosLine(zeroListDeltakapparesscr,line=1)
[zerox2kapparesscr,zeroy2kapparesscr] = extractZerosLine(zeroListDeltakapparesscr,line=2)

zeroListDeltasigmascr = findAllZeros(Deltasigma,MGrid,LambdaGrid,DeltasigmaGridscr,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1sigmascr,zeroy1sigmascr] = extractZerosLine(zeroListDeltasigmascr,line=1)
[zerox2sigmascr,zeroy2sigmascr] = extractZerosLine(zeroListDeltasigmascr,line=2)
zeroListDeltasigmaresscr = findAllZeros(Deltasigmares,MGrid,LambdaGrid,DeltasigmaGridresscr,Mp,hbar,c,e,G,Mprobe,Ms,rhos,rhop,rhobg,Rs,Rp,g0,omegam,r0,epsilon,muc,r,n,Mes,probescr)
[zerox1sigmaresscr,zeroy1sigmaresscr] = extractZerosLine(zeroListDeltasigmaresscr,line=1)
[zerox2sigmaresscr,zeroy2sigmaresscr] = extractZerosLine(zeroListDeltasigmaresscr,line=2)


# Find out which values have been mistakenly assigned to list 1. 
# j is the position in the list where the values starts being shifted

j = 0
jres = 0
k = 0
kres = 0

jscr = 0
Jresscr = 0
kscr = 0
kresscr = 0

for i in range(1, zerox1kappa.size):
	if 10**zerox1kappa[i]/10**zerox1kappa[i-1] >2:
		j = i

for i in range(1, zerox1kappares.size):
	if 10**zerox1kappares[i]/10**zerox1kappares[i-1] >2:
		jres = i

for i in range(1, zerox1sigma.size):
	if 10**zerox1sigma[i]/10**zerox1sigma[i-1] >2:
		k = i

for i in range(1,zerox1sigmares.size):
	if 10**zerox1sigmares[i]/10**zerox1sigmares[i-1] >2:
		kres = i

# For screened values

for i in range(1, zerox1kappascr.size):
	if 10**zerox1kappascr[i]/10**zerox1kappascr[i-1] >2:
		jscr = i

for i in range(1, zerox1kapparesscr.size):
	if 10**zerox1kapparesscr[i]/10**zerox1kapparesscr[i-1] >2:
		jresscr = i

for i in range(1, zerox1sigmascr.size):
	if 10**zerox1sigmascr[i]/10**zerox1sigmascr[i-1] >2:
		kscr = i

for i in range(1,zerox1sigmaresscr.size):
	if 10**zerox1sigmaresscr[i]/10**zerox1sigmaresscr[i-1] >1:
		kresscr = i



# The assign these values from list 1 to list 2
zerox2kappa = np.concatenate((zerox2kappa, zerox1kappa[j:nSample]))
zeroy2kappa = np.concatenate((zeroy2kappa, zeroy1kappa[j:nSample]))
zerox2kappares = np.concatenate((zerox2kappares, zerox1kappares[jres:nSample]))
zeroy2kappares = np.concatenate((zeroy2kappares, zeroy1kappares[jres:nSample]))

zerox2sigma = np.concatenate((zerox2sigma, zerox1sigma[k:nSample]))
zeroy2sigma = np.concatenate((zeroy2sigma, zeroy1sigma[k:nSample]))
zerox2sigmares = np.concatenate((zerox2sigmares, zerox1sigmares[kres:nSample]))
zeroy2sigmares = np.concatenate((zeroy2sigmares, zeroy1sigmares[kres:nSample]))

zerox2kappascr = np.concatenate((zerox2kappascr, zerox1kappascr[jscr:nSample]))
zeroy2kappascr = np.concatenate((zeroy2kappascr, zeroy1kappascr[jscr:nSample]))
zerox2kapparesscr = np.concatenate((zerox2kapparesscr, zerox1kapparesscr[jresscr:nSample]))
zeroy2kapparesscr = np.concatenate((zeroy2kapparesscr, zeroy1kapparesscr[jresscr:nSample]))

zerox2sigmascr = np.concatenate((zerox2sigmascr, zerox1sigmascr[kscr:nSample]))
zeroy2sigmascr = np.concatenate((zeroy2sigmascr, zeroy1sigmascr[kscr:nSample]))
zerox2sigmaresscr = np.concatenate((zerox2sigmaresscr, zerox1sigmaresscr[kresscr:nSample]))
zeroy2sigmaresscr = np.concatenate((zeroy2sigmaresscr, zeroy1sigmaresscr[kresscr:nSample]))


# Finally delete the entries from the initial array
zerox1kappa = np.delete(zerox1kappa, np.arange(j, zerox1kappa.size))
zeroy1kappa = np.delete(zeroy1kappa, np.arange(j, zeroy1kappa.size))
zerox1kappares = np.delete(zerox1kappares, np.arange(jres, zerox1kappares.size))
zeroy1kappares = np.delete(zeroy1kappares, np.arange(jres, zeroy1kappares.size))

zerox1sigma = np.delete(zerox1sigma, np.arange(k, zerox1sigma.size))
zeroy1sigma = np.delete(zeroy1sigma, np.arange(k, zeroy1sigma.size))
zerox1sigmares = np.delete(zerox1sigmares, np.arange(kres, zerox1sigmares.size))
zeroy1sigmares = np.delete(zeroy1sigmares, np.arange(kres, zeroy1sigmares.size))

zerox1kappascr = np.delete(zerox1kappascr, np.arange(jscr, zerox1kappascr.size))
zeroy1kappascr = np.delete(zeroy1kappascr, np.arange(jscr, zeroy1kappascr.size))
zerox1kapparesscr = np.delete(zerox1kapparesscr, np.arange(jresscr, zerox1kapparesscr.size))
zeroy1kapparesscr = np.delete(zeroy1kapparesscr, np.arange(jresscr, zeroy1kapparesscr.size))

zerox1sigmascr = np.delete(zerox1sigmascr, np.arange(kscr, zerox1sigmascr.size))
zeroy1sigmascr = np.delete(zeroy1sigmascr, np.arange(kscr, zeroy1sigmascr.size))
zerox1sigmaresscr = np.delete(zerox1sigmaresscr, np.arange(kresscr, zerox1sigmaresscr.size))
zeroy1sigmaresscr = np.delete(zeroy1sigmaresscr, np.arange(kresscr, zeroy1sigmaresscr.size))


# Then flip the first line and then concatenate the arrays

zerox1kappa = np.flip(zerox1kappa)
zeroy1kappa = np.flip(zeroy1kappa)
zerox1kappares = np.flip(zerox1kappares)
zeroy1kappares = np.flip(zeroy1kappares)

zerox1sigma = np.flip(zerox1sigma)
zeroy1sigma = np.flip(zeroy1sigma)
zerox1sigmares = np.flip(zerox1sigmares)
zeroy1sigmares = np.flip(zeroy1sigmares)

zerox1kappascr = np.flip(zerox1kappascr)
zeroy1kappascr = np.flip(zeroy1kappascr)
zerox1kapparesscr = np.flip(zerox1kapparesscr)
zeroy1kapparesscr = np.flip(zeroy1kapparesscr)

zerox1sigmascr = np.flip(zerox1sigmascr)
zeroy1sigmascr = np.flip(zeroy1sigmascr)
zerox1sigmaresscr = np.flip(zerox1sigmaresscr)
zeroy1sigmaresscr = np.flip(zeroy1sigmaresscr)

zeroxkappa = np.concatenate((zerox1kappa, zerox2kappa))
zeroykappa = np.concatenate((zeroy1kappa, zeroy2kappa))
zeroxkappares = np.concatenate((zerox1kappares, zerox2kappares))
zeroykappares = np.concatenate((zeroy1kappares, zeroy2kappares))

zeroxsigma = np.concatenate((zerox1sigma, zerox2sigma))
zeroysigma = np.concatenate((zeroy1sigma, zeroy2sigma))
zeroxsigmares = np.concatenate((zerox1sigmares, zerox2sigmares))
zeroysigmares = np.concatenate((zeroy1sigmares, zeroy2sigmares))

zeroxkappascr = np.concatenate((zerox1kappascr, zerox2kappascr))
zeroykappascr = np.concatenate((zeroy1kappascr, zeroy2kappascr))
zeroxkapparesscr = np.concatenate((zerox1kapparesscr, zerox2kapparesscr))
zeroykapparesscr = np.concatenate((zeroy1kapparesscr, zeroy2kapparesscr))

zeroxsigmascr = np.concatenate((zerox1sigmascr, zerox2sigmascr))
zeroysigmascr = np.concatenate((zeroy1sigmascr, zeroy2sigmascr))
zeroxsigmaresscr = np.concatenate((zerox1sigmaresscr, zerox2sigmaresscr))
zeroysigmaresscr = np.concatenate((zeroy1sigmaresscr, zeroy2sigmaresscr))


# Plot the functions
rcParams.update({'figure.autolayout': True})
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['xtick.minor.width'] = 0

fig, ax = plt.subplots(figsize = (6, 6))

sigma = plt.loglog(10**zeroxsigma,zeroysigma/e,'-', alpha = 0)
sigmares = plt.loglog(10**zeroxsigmares,zeroysigmares/e,'-', alpha = 0)

sigmascr = plt.loglog(10**zeroxsigmascr,zeroysigmascr/e,'-', alpha = 0)
sigmaresscr = plt.loglog(10**zeroxsigmaresscr,zeroysigmaresscr/e,'-', alpha = 0)

plt.loglog(MRange, Ssourceline, ':', color = 'magenta', label = '$S_S = 0$')
plt.loglog(MRange, Sprobeline, '--', color = 'orange', label = '$S_P = 0$')

plt.xlabel('$M/M_{\\mathrm{P}}$', fontfamily = 'serif',  fontsize = 15)
plt.ylabel('$\\Lambda\,(\\mathrm{eV})$', fontfamily = 'serif',  fontsize = 15)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax.set_ylim(1e-12,1e2)
ax.set_xlim(1e-15,1e2)

plt.fill_between(10**zeroxsigmares, zeroysigmares/e, 1e6,
                 alpha=1, color=(0.722869, 0.783111, 0.913125), interpolate=True, label = "$\\Delta \\sigma^{(\\mathrm{mod})}/\\sigma$")
plt.fill_between(10**zeroxsigma, zeroysigma/e, 1e6,
                 alpha=1, color=(0.603583, 0.591452, 0.910405), interpolate=True, label = "$\\Delta \\sigma/\\sigma$")
plt.fill_between(10**zeroxsigmaresscr, zeroysigmaresscr/e, 1e6,
                 alpha=1, color=(0.455659, 0.339501, 0.757464), interpolate=True, label = "$\\Delta \\sigma^{(\\mathrm{mod})}_{(\\mathrm{scr})}/\\sigma$")
plt.fill_between(10**zeroxsigmascr, zeroysigmascr/e, 1e6,
                 alpha=1, color=(0.293416, 0.0574044, 0.529412), interpolate=True, label = "$\\Delta \\sigma_{(\\mathrm{scr})}/\\sigma$")

ax.legend(loc = 'lower left', labelspacing = 0.4, fontsize = 12)

plt.savefig('ExclusionMLambdaSmallRange.pdf')
plt.show()

# Plot the convex hull

fig, ax = plt.subplots(figsize = (6, 6))
sigmares = plt.loglog(10**zeroxsigmares,zeroysigmares/e,'-', alpha = 0)
sigmaresscr = plt.loglog(10**zeroxsigmaresscr,zeroysigmaresscr/e,'-', alpha = 0)

plt.fill_between(10**zeroxsigmares, zeroysigmares/e, 1e6,
                 alpha=1, color=(0.89, 0.61, 0), interpolate=True, label = "This work")
plt.fill_between(10**zeroxsigmaresscr, zeroysigmaresscr/e, 1e6,
                 alpha=1, color=(0.96, 0.79, 0.13), interpolate=True, label = "This work (screened)")

ax.set_ylim(1e-12,1e2)
ax.set_xlim(1e-15,1e2)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax.legend(loc = 'lower left', labelspacing = 0.4, fontsize = 12)

plt.xlabel('$M/M_{\\mathrm{P}}$', fontfamily = 'serif',  fontsize = 15)
plt.ylabel('$\\Lambda\,(\\mathrm{eV})$', fontfamily = 'serif',  fontsize = 15)

plt.savefig('ExclusionLambdaM.pdf')
plt.show()