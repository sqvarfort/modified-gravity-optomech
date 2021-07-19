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

def kappafunc(Lambda,alpha,r0):
	return alpha*np.exp(- r0/Lambda)*(1 + r0/Lambda)

def sigmafunc(Lambda,alpha,r0):
	return alpha*np.exp(- r0/Lambda)*(2 + 2*r0/Lambda + r0**2/Lambda**2)

def Deltakappa(LambdaLog,alpha,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0):
	Lambda = 10**LambdaLog
	kappa = kappafunc(Lambda,alpha,r0)
	return 1/(np.sqrt(Mes)*gNewt)*(1/np.sqrt((muc**2*np.exp(4*r) + np.sinh(2*r)**2/2)))*np.sqrt(2*hbar*omegam**5/Mprobe)*(1/(8*np.pi*n*g0))*(1/kappa)-1

def Deltasigma(LambdaLog,alpha,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0):
	Lambda = 10**LambdaLog
	sigma = sigmafunc(Lambda,alpha,r0)
	return 1/(np.sqrt(Mes)*gNewt)*(1/np.sqrt((muc**2*np.exp(4*r) + np.sinh(2*r)**2/2)))*np.sqrt(2*hbar*omegam**5/Mprobe)*(1/(4*np.pi*n*g0*epsilon))*(1/sigma)-1

def Deltakappares(LambdaLog,alpha,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0):
	Lambda = 10**LambdaLog
	kappa = kappafunc(Lambda,alpha,r0)
	return 1/(np.sqrt(Mes)*gNewt)*1/np.sqrt((muc**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)*1/(4*np.pi*n*g0)*(1/kappa)-1

def Deltasigmares(LambdaLog,alpha,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0):
	Lambda = 10**LambdaLog
	sigma = sigmafunc(Lambda,alpha,r0)
	return 1/(np.sqrt(Mes)*gNewt)*1/np.sqrt((muc**2*np.exp(4*r) + np.sinh(2*r)**2/2))*np.sqrt(2*hbar*omegam**5/Mprobe)* 1/(2*np.pi**2*n**2*g0*epsilon)*(1/sigma)-1

# Function to find all zeros, given a meshgrid:
def findAllZeros(func,X,Y,Z,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0,logX=True,logY=False):
    zeroList = []
    # If the x-values are in exponential form, convert to log 
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
                xroot = optimize.brentq(func,Xuse[k,l],Xuse[k,l+1],args=(Yuse[k,l],r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0))
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

###################################

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
alphamin = float(args['alphamin'])
alphamax = float(args['alphamax'])
lmin = float(args['lmin'])
lmax = float(args['lmax'])

nSample = int(args['nSample'])

alpharange = 10**(np.linspace(alphamin,alphamax,nSample))
lambdarange = 10**(np.linspace(lmin,lmax,nSample))

LambdaGrid, alphaGrid = np.meshgrid(lambdarange,alpharange)

gNewt = G*Ms/r0**2

#test = Deltakappa(np.log10(1e-2),1e-3,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
#print(test)

DeltakappaGrid = Deltakappa(np.log10(LambdaGrid),alphaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
DeltakappaGridres = Deltakappares(np.log10(LambdaGrid),alphaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
DeltasigmaGrid = Deltasigma(np.log10(LambdaGrid),alphaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
DeltasigmaGridres = Deltasigmares(np.log10(LambdaGrid),alphaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)

zeroListDeltakappa = findAllZeros(Deltakappa,LambdaGrid,alphaGrid,DeltakappaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
[zerox1kappa,zeroy1kappa] = extractZerosLine(zeroListDeltakappa,line=1)
[zerox2kappa,zeroy2kappa] = extractZerosLine(zeroListDeltakappa,line=2)

zeroListDeltakappares = findAllZeros(Deltakappares,LambdaGrid,alphaGrid,DeltakappaGridres,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
[zerox1kappares,zeroy1kappares] = extractZerosLine(zeroListDeltakappares,line=1)
[zerox2kappares,zeroy2kappares] = extractZerosLine(zeroListDeltakappares,line=2)

zeroListDeltasigma = findAllZeros(Deltasigma,LambdaGrid,alphaGrid,DeltasigmaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
[zerox1sigma,zeroy1sigma] = extractZerosLine(zeroListDeltasigma,line=1)
[zerox2sigma,zeroy2sigma] = extractZerosLine(zeroListDeltasigma,line=2)

zeroListDeltasigmares = findAllZeros(Deltasigmares,LambdaGrid,alphaGrid,DeltasigmaGridres,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0)
[zerox1sigmares,zeroy1sigmares] = extractZerosLine(zeroListDeltasigmares,line=1)
[zerox2sigmares,zeroy2sigmares] = extractZerosLine(zeroListDeltasigmares,line=2)

# Flip arrays to make them easier to work with 

zerox1kappa = np.flip(zerox1kappa)
zeroy1kappa = np.flip(zeroy1kappa)
zerox1kappares = np.flip(zerox1kappares)
zeroy1kappares = np.flip(zeroy1kappares)

zerox1sigma = np.flip(zerox1sigma)
zeroy1sigma = np.flip(zeroy1sigma)
zerox1sigmares = np.flip(zerox1sigmares)
zeroy1sigmares = np.flip(zeroy1sigmares)

# Add values for the horizontal lines which cannot be found with the zero-finding function

h = 0.1

for i in range(0, 100):
	zerox1kappa = np.append(zerox1kappa, zerox1kappa[-1]+h)
	zeroy1kappa = np.append(zeroy1kappa, zeroy1kappa[-1])
	zerox1kappares = np.append(zerox1kappares, zerox1kappares[-1]+h)
	zeroy1kappares = np.append(zeroy1kappares, zeroy1kappares[-1])
	zerox1sigma = np.append(zerox1sigma, zerox1sigma[-1]+h)
	zeroy1sigma = np.append(zeroy1sigma, zeroy1sigma[-1])
	zerox1sigmares = np.append(zerox1sigmares, zerox1sigmares[-1]+h)
	zeroy1sigmares = np.append(zeroy1sigmares, zeroy1sigmares[-1])

# Plot the functions

rcParams.update({'figure.autolayout': True})
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['xtick.minor.width'] = 0

fig, ax = plt.subplots(figsize = (6, 6))

plt.loglog(10**zerox1kappa, zeroy1kappa,'--',alpha = 1, color = 'darkgreen', label = '$\\Delta \\kappa/\\kappa$')
plt.loglog(10**zerox1kappares, zeroy1kappares,':', alpha = 1, color = 'green', label = '$\\Delta \\kappa^{(\\mathrm{mod})}/\\kappa$')
plt.loglog(10**zerox1sigma, zeroy1sigma, alpha = 0)
plt.loglog(10**zerox1sigmares, zeroy1sigmares, alpha = 0)

plt.xlabel('$\\lambda \, (\\mathrm{m})$', fontfamily = 'serif', fontsize = 15)
plt.ylabel('$|\\alpha|$', fontfamily = 'serif', fontsize = 15)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax.set_ylim(1e-8,1e8)
ax.set_xlim(1e-5,1)

plt.fill_between(10**zerox1sigmares, zeroy1sigmares, 1e10,
                 alpha=1, color=(0.455659, 0.339501, 0.757464), interpolate=True, label = "$\\Delta \\sigma^{(\\mathrm{mod})}/\\sigma$")

plt.fill_between(10**zerox1sigma, zeroy1sigma, 1e10,
                 alpha=1, color=(0.293416, 0.0574044, 0.529412), interpolate=True, label = "$\\Delta \\sigma/\\sigma$")

ax.legend(loc = 'lower left', labelspacing = 0.4, fontsize = 12)

plt.savefig('alphalambdasensitivity.pdf')
plt.show()

# Plot the convex hull

fig, ax = plt.subplots(figsize = (6, 6))

sigmares = plt.loglog(10**zerox1sigmares,zeroy1sigmares, alpha = 0)

plt.fill_between(10**zerox1sigmares, zeroy1sigmares, 1e10,
                 alpha=1, color=(0.96, 0.79, 0.13), interpolate=True, label = "This work")

ax.set_ylim(1e-8,1e8)
ax.set_xlim(1e-5,1)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.xlabel('$\\lambda \, (\\mathrm{m})$', fontfamily = 'serif',  fontsize = 15)
plt.ylabel('$|\\alpha|$', fontfamily = 'serif',  fontsize = 15)

ax.legend(loc = 'lower left', labelspacing = 0.4, fontsize = 12)
plt.savefig('Exclusionalphalambda.pdf')
plt.show()