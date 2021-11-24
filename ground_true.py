# -*- coding: utf-8 -*-
"""
Created on Sat Jul 17 09:32:10 2021

@author: User
"""

from ADO_Experiment import *
import random
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import statistics
from priors import *


expan = 1000
n = 10
k = 100
aux = np.linspace(0, expan, n)
prior = Prior()
p = prior.set_prior_exp(expan,n,k)

exp = Experiment()
exp.generate(n,k)
exp.set_prior(p)
p_prior = exp.p.copy()
mat_prior = exp.mat.copy()
invmat_prior = exp.invmat.copy()

true = 1 - 0.9* np.exp(-aux*0.005) #set grounf true

# plot prior and ground true
plt.figure(1)
plot = exp.p.T
plot[plot<1e-10] = 1e-10
plot = np.log10(plot)
plt.imshow(plot, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
# plt.imshow(exp.p.T, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
plt.scatter(np.linspace(0 + 0.5, n-0.5,n)*expan/n,1 - true ,s = 100,c='blue')
plt.plot(np.linspace(0 + 0.5, n-0.5, n)*expan/n,1 - true ,c='blue')
plt.xlabel('design (time, offer, etc.) (a.u.)', fontsize=14)
plt.ylabel('probability', fontsize=14)
plt.title("likelihood prior and ground true")


trials = 300
ind = 20 # number of runs used to average the MSE vs. trial curve
xs = np.zeros(shape=(trials, 2,ind))
MSEs = np.zeros(shape=(trials,2,ind))

for jj in range(2):
    for ii in range(ind):
        print("run number:",ii)
        exp.reset()
        # exp.generate(n,dp)
        exp.set_prior_mat(p_prior, mat_prior, invmat_prior)                
        x = []
        y = []
        MSE = []
        for i in range(trials):  
            if jj == 0:
                aux1 = exp.ADOchoose()
            else:
                aux1 = exp.RANDchoose()    
            aux2 = random.random()
            if aux2 <= true[aux1]:
                exp.update(aux1,1)
            else:
                exp.update(aux1,0)                
            x.append(i)
            estimated = exp.values*exp.p
            estimated = estimated.sum(axis = 1)
            err = estimated - true
            aux_err = np.sqrt(np.mean(err*err)/k)
            MSE.append(aux_err)
            MSE_aux = exp.values*exp.p
            
        MSEs[:,jj,ii] = MSE
        xs[:,jj,ii] = x


colors = ["green", "blue",  "orange", "red", "cyan", "pink", "olive"]
stratName = ["ADO","Random"]

plt.figure(2)
for i in range(2):
    # plt.errorbar(xs[:,i].mean(axis = 1), MSEs[:,i].mean(axis = 1), MSEs[:,i].std(axis = 1)/np.sqrt(ind), label=stratName[i],c=colors[i])
    plt.scatter(xs[:,i].mean(axis = 1), np.log10(MSEs[:,i].mean(axis = 1)), label=stratName[i],c=colors[i])
    plt.xlabel('Trial', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    

plt.figure(3)
plot = exp.p.T
plot[plot<1e-10] = 1e-10
plot = np.log10(plot)
plt.imshow(plot, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
# plt.imshow(exp.p.T, cmap='hot',origin='upper', extent=[0,expan,0,1], aspect="auto")
plt.scatter(np.linspace(0 + 0.5, n-0.5,n)*expan/n,1 - true ,s = 100,c='blue')
plt.plot(np.linspace(0 + 0.5, n-0.5, n)*expan/n,1 - true ,c='blue')
plt.xlabel('design (time, offer, etc.) (a.u.)', fontsize=14)
plt.ylabel('probability', fontsize=14)
plt.title("likelihood posterior and ground true")

