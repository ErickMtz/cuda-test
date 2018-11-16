#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:47:35 2018

@author: erick
"""
import numpy as np     
from timeit import default_timer as timer
from numba import guvectorize, float64, int64
import math
#
#@guvectorize([(float64[:],float64[:],float64[:])], '(N),(M)->(N)', target='cpu', nopython=True)
#def VectorAdd(a,b,c):
#    c = a + b[0:-1]
    
@guvectorize([(float64[:,:], float64[:], float64[:,:], int64, float64, float64[:,:])], '(K,NT),(Nc),(M,N),(),()->(M,Nc)', target='cuda', nopython=True)
def calculateCinMinibatch(W,auxi,v,n,B,c):
    M,N = v.shape
    K,Nc = W.shape[0],(W.shape[1]-N)
    aux = 0
    for A in range(M):
        for alpha in range(Nc):
            aux2 = 0
            for miu in range(K):
                aux1 = 0
                for i in range(N):
                    if W[miu][i] * v[A][i] >= 0:
                        aux += math.pow(W[miu][i] * v[A][i], n)
                    else: 
                        aux = 0
                aux2 += W[miu][N+alpha] * aux1
            c[A][alpha] = math.tanh(B*aux2)


def calculateCinMinibatch2(W,auxi,v,n,B,c):
    M,N = v.shape
    K,Nc = W.shape[0],(W.shape[1]-N)
    for A in range(M):
        aux = np.dot(W[0:K,0:N], np.transpose(v[A]))
        for i in range(len(aux)):
            if aux[i] >= 0:    
                aux[i] = math.pow(aux[i], n)
            else:
                aux[i] = 0
        c[A] = B * np.sum(W[0:K,N:N+Nc] * (aux).reshape(K,1),axis=0)
        for i in range(Nc):
            c[A][i] = math.tanh(c[A][i])


def main():
    M = 1000
    K = 2000
    N = 28*28
    Nc = 10
    n = 30
    B = 1/math.pow(650,n)
    W = np.ones((K,N+Nc), dtype=np.float64)
    v = np.ones((M,N),dtype=np.float64)
    print(W.shape)
    c = np.ones((M,Nc),dtype=np.float64)
    start = timer()
    calculateCinMinibatch(W,W[0,N:N+Nc],v,n,B,c)
    vectoradd_time = timer() - start
    
    print("c ", c)
    
    print("VectorAdd took %f seconds" % vectoradd_time)

if __name__ == '__main__':
    main()
