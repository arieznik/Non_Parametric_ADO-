from ast import ExtSlice
import math
import random   
import numpy as np
from numpy.lib.function_base import median
import scipy.stats

eps = 1e-100  # this is used to avoid log(0)

class Experiment:

    def __init__(self):
        self.n = 10
        self.k = 10
        k = self.k
        n = self.n
        minSize = 1/(k - 1)
        v = [i*minSize for i in range(math.ceil(1/minSize)+1)]
        v[0] = 0.01
        v[-1] = 0.99
       
        p = np.zeros((n,k)) 
        x = np.ones(k)
        xx = np.zeros((k,k))
        tri = np.triu(np.ones((k,k)))

        for i in range(n):
            p[i] = x
            x = x @ tri

        self.factor = p
        self.ffactor = np.flip(p)

        p = self.ffactor * p # numbers of models passing by each point n,k
        self.modelsperpointlast = p[n-1]

        # total models

        self.m = p[n-1].sum()
                
        # normalize
        p = p / p.sum(axis=1)[:,None]
        

        self.p = p
        self.hperpoint = p * self.m
        self.k = k
        self.n = n
        self.values = np.array(v)
        self.mat, self.invmat = self.calcFullFactorMatrices()
        self.initialEntropy = (np.log2(self.m/self.hperpoint) * self.p).sum()
        

    def normalize(v):
        sum = v.sum()
        if sum == 0:
            return v

        return v/sum

    def createPredictor(self): # calculate marginal probabilty for 1
        return self.p @ self.values

    def reset(self): # reset the likelihood values considering all monotonic curves equally probable
        self.p = self.factor * self.ffactor
        self.p = self.p / self.p.sum(axis=1)[:,None]
        # self.p = np.ones((self.n,self.k))/self.k 

        self.mat, self.invmat = self.calcFullFactorMatrices()
    
    def set_prior_mat(self, p, mat, invmat): # if you saved the p and M matrix use this function
        self.p = p.copy()
        self.mat = mat.copy()
        self.invmat = invmat.copy()
        
    def set_prior(self,p): # if you give a p matrix, this function calculates the corresponding M matrices
        ab_n = 400
        k = self.k
        n = self.n
        
        ind_points = 1000
        values_points = np.linspace(0.8, 0.99, ind_points)
        for i in range(ind_points):
            for a in range(n):
                v = [1]*k
                for j in range(k):
                    if self.p[a][j] > p[a][j]:
                        v[j] = values_points[i]
                # print(v)
                self.bayesUpdate(a,v)

        
    def generate(self, n, k): # it generates all the Experiment parameters using n desings and k likelihoods
        minSize = 1/(k - 1)
        v = [i*minSize for i in range(math.ceil(1/minSize)+1)]
        v[0] = 0.01
        v[-1] = 0.99
       
        p = np.zeros((n,k)) 
        x = np.ones(k)
        xx = np.zeros((k,k))
        tri = np.triu(np.ones((k,k)))

        for i in range(n):
            p[i] = x
            x = x @ tri

        self.factor = p
        self.ffactor = np.flip(p)

        p = self.ffactor * p # numbers of models passing by each point n,k
        self.modelsperpointlast = p[n-1]

        # total models

        self.m = p[n-1].sum()
                
        # normalize
        p = p / p.sum(axis=1)[:,None]
        

        self.p = p
        self.hperpoint = p * self.m
        self.k = k
        self.n = n
        self.values = np.array(v)
        self.mat, self.invmat = self.calcFullFactorMatrices()
        self.initialEntropy = (np.log2(self.m/self.hperpoint) * self.p).sum()
        
    def bayesUpdate(self, amount, v):
        aux = self.p[amount].copy() 
        self.p[amount] *= v
        self.p[amount] /= self.p[amount].sum()
        v = self.p[amount]/aux # used in forwad propagation
        v[aux<eps]=0
        vv = v # used in backward propagation

        for i in range(amount + 1, self.n):
            aux = self.p[i].copy()
            self.mat[i-1] *= v[np.newaxis, :].T
            self.p[i] = self.mat[i-1].sum(axis = 0)
            v = self.p[i]/aux
            v[aux<eps]=0

        for i in range(amount-1, -1, -1):
            aux = self.p[i].copy()
            self.mat[i] *= vv
            self.p[i] = self.mat[i].sum(axis = 1)
            vv = self.p[i]/aux
            vv[aux<eps]=0

    def update(self, amount, result):
        v = self.values
        if result == 0:
            v = 1-v

        self.bayesUpdate(amount, v)
        
        
    def calcFullFactorMatrices(self):
        mat = []
        invmat = []

        for i in range(self.n-1):
            m = self.calcFactorMatrix(i,i+1)
            mat.append(m)
            invmat.append(np.linalg.inv(m))

        # return mat,invmat
        return np.array(mat),np.array(invmat)

    def calcFactorMatrix(self, i, j):
        ret = np.zeros((self.k, self.k))

        for a in range(self.k):
            for b in range(self.k):
                ret[a,b] = self.calcFactor(i,a,j,b)

        return ret/ret.sum()    

    def calcFactor(self, x0, y0, x1, y1):
        if x1 < x0:
            x1,x0 = x0,x1
            y1,y0 = y0,y1

        if y1<y0: return 0

        dx = x1-x0
        dy = y1-y0

        if dx == 0:
            if dy == 0:
                center = 1
            else:
                center = 0
        else:
            center = (self.factor[dx-1, dy]) 

        return self.factor[x0,y0] * center * self.ffactor[x1,y1]
        
    def getRandomModel(self): # it takes a random model from the p and M matrices
        groundtruth = np.zeros(self.n)
        aux = 0
        for i in range(0,self.n - 1):
            aux = random.choices(np.arange(aux, self.k,1), weights=self.mat[i].sum(axis =1)[aux:self.k])
            aux = aux[0]
            groundtruth[i] = aux
        
        aux = random.choices(np.arange(aux, self.k,1), weights=self.mat[i].sum(axis =1)[aux:self.k])
        aux = aux[0]
        groundtruth[i + 1] = aux
        
        return groundtruth
   
    def bayesCalcExpectedGainAll(self, amount, v): # calculate the expected info gain at a given design value       
        aux = self.p[amount].copy()
        p = self.p.copy()
        mat = self.mat.copy()
        p[amount] *= v
        p[amount] /= p[amount].sum()
        v = p[amount]/aux # used in forwad propagation
        v[aux<eps]=0
        vv = v # used in backward propagation

        for i in range(amount + 1, self.n):
            aux = self.p[i].copy()
            mat[i-1] *= v[np.newaxis, :].T
            p[i] = mat[i-1].sum(axis = 0)
            v = p[i]/aux
            v[aux<eps]=0

        for i in range(amount-1, -1, -1):
            aux = self.p[i].copy()
            mat[i] *= vv
            p[i] = mat[i].sum(axis = 1)
            vv = p[i]/aux
            vv[aux<eps]=0
        
        return p
    
    
    def CalcExpectedGainAll(self): # calculate the expected gain at each design point
        gain = np.zeros(self.n)
        p1 = self.createPredictor()
        p0 = 1-p1
        for i in range(self.n):
            v = self.values.copy()
            info1 = self.bayesCalcExpectedGainAll(i, v)  
            v = 1-v
            info0 = self.bayesCalcExpectedGainAll(i, v)
    
            info1 = np.log2(info1/self.p)*info1
            info1[self.p<eps] = 0
            info1 = info1.sum()
            
            info0 = np.log2(info0/self.p)*info0
            info0[self.p<eps] = 0
            info0 = info0.sum()
        
            gain[i] = p1[i]*info1 + p0[i]*info0
        
        return gain
    
    def ADOchoose(self): # choses the design wich maximizes the expected gain
        a =  np.argmax(self.CalcExpectedGainAll())
        return a
    
    def RANDchoose(self): # choses a random design value
        a = random.randint(0,self.n-1)
        return a
