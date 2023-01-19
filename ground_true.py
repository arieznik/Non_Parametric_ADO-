
from ADO_Experiment import *

import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from priors import *

expan = 1000
n = 10
k = 100

exp = Experiment()
exp.generate(n,k)
aux = np.linspace(0, expan, n)

prior = Prior()
p = prior.set_prior_exp(expan,n,k)
exp.set_prior(p)


p_prior = exp.p.copy() 
mat_prior = exp.mat.copy()
invmat_prior = exp.invmat.copy()

true = 1 - 0.9* np.exp(-aux*0.005) #exp

# plot prior and ground true
plt.figure(1)
plot = exp.p.T
plot[plot<1e-10] = 1e-10
plot = np.log10(plot)
plt.imshow(plot, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
plt.scatter(np.linspace(0 + 0.5, n-0.5,n)*expan/n,1 - true ,s = 100,c='blue')
plt.plot(np.linspace(0 + 0.5, n-0.5, n)*expan/n,1 - true ,c='blue')
plt.xlabel('design (time, offer, etc.) (a.u.)', fontsize=14)
plt.ylabel('probability', fontsize=14)
plt.title("likelihood prior and ground true")


trials = 300
ind =  320# number of runs used to average the MSE vs. trial curve
xs = np.zeros(shape=(trials+1, 2,ind))
MSEs = np.zeros(shape=(trials+1,2,ind))
infoGained = np.zeros(shape=(trials+1,2,ind))
designs = []
results = []

for jj in range(2):
    for ii in range(ind):
        print("run number:",ii)
        exp.reset()
        exp.set_prior_mat(p_prior, mat_prior, invmat_prior)
        
        x = [0]
        y = []
        estimated = exp.values*exp.p
        estimated = estimated.sum(axis = 1)
        err = estimated - true
        aux_err = np.mean(err*err)
        MSE = [aux_err]
        info = [exp.infoProgress()]
        for i in range(trials):  
            if jj == 0:
                aux1 = exp.ADOchoose()  
                designs.append(aux1)
            else:
                aux1 = exp.RANDchoose()    
            aux2 = random.random()
            if aux2 <= true[aux1]:
                exp.update(aux1,1)
            else:
                exp.update(aux1,0)  
            results.append(aux2 <= true[aux1])
            x.append(i+1)
            estimated = exp.values*exp.p
            estimated = estimated.sum(axis = 1)
            err = estimated - true
            aux_err = np.mean(err*err)
            MSE.append(aux_err)
            info.append(exp.infoProgress())
        aux1 = exp.ADOchoose()  
        designs.append(aux1)    
        MSEs[:,jj,ii] = MSE
        xs[:,jj,ii] = x
        infoGained[:,jj,ii] = info
 

colors = ["green", "blue"]
stratName = ["ADO","Random"]



plt.figure(2)
for i in range(2):
    plt.errorbar(xs[:,i].mean(axis = 1), np.log10(MSEs[:,i].mean(axis = 1)), yerr=0.434* MSEs[:,i].std(axis = 1)/MSEs[:,i].mean(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    
plt.figure(20)
for i in range(2):
    plt.errorbar(xs[:,i].mean(axis = 1), infoGained[:,i].mean(axis = 1), yerr=infoGained[:,i].std(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('InfoGained', fontsize=14)

plt.figure(4) 
plot = exp.p.T
plot[plot<1e-10] = 1e-10
plot = np.log10(plot)
plt.imshow(plot, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
plt.scatter(np.linspace(0 + 0.5, n-0.5,n)*expan/n,1 - true ,s = 100,c='blue')
plt.plot(np.linspace(0 + 0.5, n-0.5, n)*expan/n,1 - true ,c='blue')
plt.xlabel('design (time, offer, etc.) (a.u.)', fontsize=14)
plt.ylabel('probability', fontsize=14)
