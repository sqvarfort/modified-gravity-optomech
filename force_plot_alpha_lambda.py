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
	FN = gNewt*Mprobe
	return gNewt*Mprobe*kappa/FN

def Deltasigmares(Lambda,alpha,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0):
	sigma = sigmafunc(Lambda,alpha,r0)
	FN = gNewt*Mprobe
	return gNewt*epsilon*Mprobe*sigma/FN

# Function to find all zeros, given a meshgrid:
def findAllZeros(func,X,Y,Z,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0,bound,logX=True,logY=False):
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
				xroot = optimize.brentq(func,Xuse[k,l],Xuse[k,l+1],args=(Yuse[k,l],r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0,bound))
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

gNewt = G*Ms/r0**2

alpharange = 10**(np.linspace(alphamin,alphamax,nSample))
lambdarange = 10**(np.linspace(lmin,lmax,nSample))

LambdaGrid, alphaGrid = np.meshgrid(lambdarange,alpharange)

# Plot the functions
rcParams.update({'figure.autolayout': True})
plt.rcParams['xtick.minor.size'] = 0
plt.rcParams['xtick.minor.width'] = 0
fig, ax = plt.subplots(figsize = (7, 6))
plt.xlabel('$\\lambda \, (\\mathrm{m})$', fontfamily = 'serif', fontsize = 15)
plt.ylabel('$|\\alpha|$', fontfamily = 'serif', fontsize = 15)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.set_ylim(1e-8,1e8)
ax.set_xlim(1e-5,1)

# Start loop that prints each bound in bounds

DeltakappaGrid = np.log10(Deltakappares(LambdaGrid,alphaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0))
DeltasigmaGrid = np.log10(Deltasigmares(LambdaGrid,alphaGrid,r0,gNewt,muc,r,hbar,omegam,Mprobe,n,g0))

#plt.loglog(10**zerox1sigmares, zeroy1sigmares, alpha = 1, color = 'black', label = str(bound))

viridis = cm.get_cmap('viridis', 8)
#loglevels = [1e-20,1e-18,1e-16,1e-14,1e-12,1e-10,1e-1]
#levels = [-20, -18, -16, -14, -12, -10, -8, -6]
levels = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
CS = ax.contourf(LambdaGrid, alphaGrid, DeltasigmaGrid, levels = levels, colors = viridis.colors)
plt.xscale('log')
plt.yscale('log')
clb = fig.colorbar(CS, extend = 'both')
clb.ax.tick_params(labelsize=12)
clb.set_label('$\\log_{10}(\\Delta F)$', labelpad=-40, y=1.05, rotation=0, fontsize = 12)
#ax.legend(loc = 'lower left', labelspacing = 0.4, fontsize = 12)
#ax.clabel(CS, inline=True, fontsize=10)
plt.savefig('alphalambdaforceplot.pdf')
plt.show()
