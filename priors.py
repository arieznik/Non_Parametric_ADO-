# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 09:30:32 2021

@author: test
"""
from ast import ExtSlice
import math
import random   
import numpy as np
from numpy.lib.function_base import median
import scipy.stats
# from model2 import *
import random

class Prior:
    def __init__(self):
        self.n = 10
        self.k = 10

    def set_prior_exp(self,expan,n,k):
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 400 # this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        a_grid = np.linspace(0, 1, ab_n)
        dist_a = scipy.stats.beta(2, 1)
        a_prob = dist_a.pdf(a_grid)/ab_n
        
        b_grid = np.linspace(0, 1, ab_n)
        dist_b = scipy.stats.beta(1, 80)
        b_prob = dist_b.pdf(b_grid)/ab_n
        for i in range(ab_n):
            for ii in range(ab_n):
                true = (k-1) - (k-1)*  ( a_grid[i]* np.exp(-b_grid[ii]*aux ) )                                   
                for iii in range(n):
                    p[iii,round(true[iii])] += a_prob[i]*b_prob[ii]
        
        p = n * p/p.sum()
        
        return p
    
    
    def set_prior_pow(self,expan,n,k):
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 400# this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        a_grid = np.linspace(0, 1, ab_n)
        dist_a = scipy.stats.beta(2, 1)
        a_prob = dist_a.pdf(a_grid)/ab_n
        
        b_grid = np.linspace(0, 1, ab_n)
        dist_b = scipy.stats.beta(1, 4)
        b_prob = dist_b.pdf(b_grid)/ab_n
        for i in range(ab_n):
            for ii in range(ab_n):
                true = (k-1) - (k-1)*(a_grid[i]*(aux + 1)**-b_grid[ii])
                for iii in range(n):
                    p[iii,round(true[iii])] += a_prob[i]*b_prob[ii]
        
        p = n * p/p.sum()
        
        return p      

    def set_prior_logis(self,expan,n,k):
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 40000# this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        for i in range(ab_n):
            true = (k-1)/(1 + np.exp(-0.1*random.random()*(aux-200 +  100*random.random())))
            for iii in range(n):
                p[iii,round(true[iii])] += 1/ab_n        
        p = n * p/p.sum()
        
        return p  
    
    
    def set_prior_cuadra(self,expan,n,k):
        self.k = k
        self.n = n
        p = np.zeros((n,k))
        ab_n = 40000# this  numeber is arbitrary and based on convergence test we did 
        # using n =10 and k =100. If you increase these number be sure you achieved convergence
        aux = np.linspace(0, expan, n)
        
        for i in range(ab_n):
            # true = (k-1)*random.random() + (k-1)*random.random()*(aux/expan)**2
            true = (k-1)*random.random() + (k-1)*random.random()*(2*aux/expan)**2
            true[true>(k-1)] = k-1
            for iii in range(n):
                p[iii,round(true[iii])] += 1/ab_n        
        p = n * p/p.sum()
        
        return p  
    
    
    def set_prior_gauss(self,expan,n,k):
        p = np.zeros((n,k))
        aux = np.linspace(0, expan, n)
        means = (k/2) * (1 - aux/800)
        x = np.linspace(1, k, k)
        
        for i in range(n):
            p[i,:] = 1./((k/10)* math.sqrt(2*math.pi))*np.exp(-(0.5/((k/10)**2))*np.power((x - means[i]), 2.))
            # p[i,:] = gaussian(x, k/10, means[i])
            p[i,:] = np.flip(p[i,:]/p[i,:].sum())
        
        return p 