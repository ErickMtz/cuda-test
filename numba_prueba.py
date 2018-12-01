
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:47:35 2018
@author: erick
"""
import numpy as np     
import time
from numba import jit, prange
import math
import os
import matplotlib.pyplot as plt

@jit(nopython=True, parallel=True, nogil=True)
def calculateCinMinibatchJit(W,v,n,B,c):
    M,N = v.shape
    K,Nc = W.shape[0],(W.shape[1]-N)
    for A in prange(M):
        aux = np.dot(W[0:K,0:N], np.transpose(v[A]))
        for i in prange(len(aux)):
            if aux[i] >= 0:    
                aux[i] = np.power(aux[i], float(n-1))
            else:
                aux[i] = 0
        c[A] = np.tanh(B * np.sum(W[0:K,N:N+Nc] * aux.reshape(K,1),axis=0))


@jit(nopython=True, nogil=True)
def calculatedWinMinibatchJit2(c,tX,m,W,v,B,n,dW):
    M,N = v.shape
    K,Nc = W.shape[0],(W.shape[1]-N)
    aux1 = np.power(c - tX, float(2*m-1)) * (1 - np.power(c, float(2)))
    if np.any(np.isinf(aux1)):
        print("Error (inf): fun: calculatedWinMinibathJit2, aux1 result is inf")
    if np.any(np.isnan(aux1)):
        print("Error (nan): fun: calculatedWinMinibathJit2, aux1 result is nan")
    for miu in prange(K):
        aux2 = np.dot(W[miu,0:N], np.transpose(v))
        if np.any(np.isinf(aux2)):
            print("Error (inf): fun: calculatedWinMinibathJit2, aux2 result is inf")
        if np.any(np.isnan(aux2)):
            print("Error (nan): fun: calculatedWinMinibathJit2, aux2 result is nan")
        for i in prange(len(aux2)):
            if aux2[i] >= 0:    
                aux2[i] = np.power(aux2[i], float(n-2))
            else:
                aux2[i] = 0.0
        dW[miu,0:N] = 2*m*B*n*np.sum(np.sum(aux1 * W[miu,N:N+Nc] * aux2.reshape(M,1),axis=1).reshape(M,1) * v, axis=0)
        dW[miu,N:N+Nc]= 2*m*B*np.sum(aux1 * (aux2*np.dot(W[miu,0:N], np.transpose(v))).reshape(M,1), axis=0)
        if np.any(np.isinf(dW)):
            print("Error (inf): fun: calculatedWinMinibathJit2, dW result is inf")
        if np.any(np.isnan(dW)):
            print("Error (nan): fun: calculatedWinMinibathJit2, dW result is nan")


@jit(nopython=True, parallel=True, nogil=True)
def updateW(V,dW,e,p,Nc,W):
    K,N = W.shape[0],(W.shape[1]-Nc)
    for miu in prange(K):
        for I in prange(N+Nc):
            V[miu][I] = p*V[miu][I] - dW[miu][I]
            if np.any(np.isinf(V[miu])):
                print("Error (inf): fun: updateW, V result is inf")
            if np.any(np.isnan(V[miu])):
                print("Error (nan): fun: updateW, V result is nan")
            W[miu][I] = W[miu][I] + (e*V[miu][I]/np.max(np.abs(V[miu])))  
            if np.any(np.isinf(W[miu])):
                print("Error (inf): fun: updateW, W result is inf")
            if np.any(np.isnan(W[miu])):
                print("Error (nan): fun: updateW, W result is nan")
            if W[miu][I] < -1:
                W[miu][I] = -1.0
            elif W[miu][I] > 1:
                W[miu][I] = 1.0
                        


try:
    prior = os.nice(-20)
    print(prior)
except OSError as err:
    print(err)

n = 30
T = 650
m = 29

K = 2000 #Number of memories
p = 0.9 # 0.6 <= p >= 0.95 Momentum

B = 1/math.pow(T,n)
f = np.vectorize(lambda x:0 if x<0 else np.float_power(x,n), otypes=[float])    


# Read digits from MNIST database
from mnist import MNIST
mndata = MNIST('samples')
images, labels = mndata.load_training()
height = int(math.sqrt(len(images[0])))
width = int(math.sqrt(len(images[0])))
N = len(images[0])

z = [((np.asarray(x)-127.5)/127.5).tolist() for _, x in sorted(zip(labels, images))]
nImages = [labels.count(x) for x in sorted(set(labels))]


nExamplesPerClass = 5000
imagesPerClassInMinibatch = 100
Nc = 10


X = np.array([z[x:x+nExamplesPerClass] for x in np.cumsum([0]+nImages[0:-1])]) #Training set
Te = np.array([z[x+5000:y] for x,y in zip(np.cumsum([0]+nImages[0:-1]),np.cumsum(nImages))]) #Testing set

tX = np.array(sorted((np.identity(Nc)*2-1).tolist() * imagesPerClassInMinibatch, reverse=True)) #Training labels set
#tT = sorted(np.repeat(np.identity(10, dtype=np.int), repeats = [n-5000 for n in nImages], axis=0).tolist(), reverse=True) #Testing labels set

# Training
W = np.random.normal(-0.3, 0.3, (K, N+Nc))
V = np.zeros((K,N+Nc))#np.concatenate((X[:, np.random.choice(nExamplesPerClass, int(K/Nc), replace=False)].reshape(K,N), np.array(sorted((np.identity(Nc)*2-1).tolist() * int(K/Nc), reverse=True))),axis=1)
np.random.shuffle(V)


nEpochs = 300

eo = 0.01 # 0.01<= eo >= 0.04
fe = 0.998

M = imagesPerClassInMinibatch*Nc #Minibatch size
nUpdates = int(len(X)*len(X[0])/M)
print(nEpochs, nUpdates)    

error_training = np.zeros((nEpochs))+100
error_testing = np.zeros((nEpochs))+100
error_obj = np.zeros((nEpochs))+100
memoriestograph = np.random.randint(K,size=25)


for epoch in range(nEpochs):
    e = eo*np.float_power(fe,epoch) 
    obj_func = 0
    start = time.time()
    for t in range(nUpdates):
        
        #print("epoch =", epoch+1, ",update =",t+1)
        c = np.zeros((M,Nc))
        dW = np.zeros((K,N+Nc))
        v = np.array([i[t*imagesPerClassInMinibatch:t*imagesPerClassInMinibatch+imagesPerClassInMinibatch] for i in X]).reshape(M, N)
        
        ## MINIBATCH
        
        calculateCinMinibatchJit(W,v,n,B,c)
        calculatedWinMinibatchJit2(c,tX,m,W,v,B,n,dW)
        updateW(V,dW,e,p,Nc,W)
        obj_func += np.sum(np.power(c-tX,2*m))
        
        #print("Minibatch time %s" % (end - start))
    
    
    
#    confusion_matrix = np.zeros((Nc,Nc), dtype=int)
#    for i in prange(Nc):
#        for j in prange(100):
#            tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
#            c = np.zeros(Nc) - 1 
#            for alpha in prange(Nc):
#                tsn[alpha] = 1
#                eng1 = np.sum(f(np.dot(np.concatenate((X[i][j], tsn), axis=0), np.transpose(W))))
#                tsn[alpha] = -1
#                eng2 = np.sum(f(np.dot(np.concatenate((X[i][j], tsn), axis=0), np.transpose(W))))
#                c[alpha] = np.tanh(B*(eng1-eng2))
#            confusion_matrix[i][np.argmax(c)] += 1
#    print("training error",(1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100)
#    error_training[epoch] = (1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100
#    
#    
    confusion_matrix = np.zeros((Nc,Nc), dtype=int)
    for i in prange(Nc):
        for j in prange(len(Te[i])):
            tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
            c = np.zeros(Nc) - 1 
            for alpha in prange(Nc):
                tsn[alpha] = 1
                eng1 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
                tsn[alpha] = -1
                eng2 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
                c[alpha] = np.tanh(B*(eng1-eng2))
            confusion_matrix[i][np.argmax(c)] += 1
    print("testing error",(1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100)
    error_testing[epoch] = (1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100
        
    error_obj[epoch] = obj_func
    print("time",start - time.time(),"epoch=",epoch,"obj=",obj_func)
    
    for i in memoriestograph:
        plt.imsave(str(i)+"epoch"+str(epoch),W[i,0:N].reshape(width, height), vmin=-1, vmax=1,  cmap="coolwarm", format="png")
     
     
np.savetxt('error_training.txt', error_training, delimiter=",")
np.savetxt('error_testing.txt', error_testing, delimiter=",")
np.savetxt('error_obj.txt', error_obj, delimiter=",")
np.savetxt('W.txt', W, delimiter=",")
print("END TRAINING")


confusion_matrix = np.zeros((Nc,Nc), dtype=int)
for i in range(Nc):
    for j in range(len(Te[i])):
        tsn = np.array([-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1.,-1])
        c = np.zeros(Nc) - 1 
        for alpha in prange(Nc):
            tsn[alpha] = 1
            eng1 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
            tsn[alpha] = -1
            eng2 = np.sum(f(np.dot(np.concatenate((Te[i][j], tsn), axis=0), np.transpose(W))))
            c[alpha] = np.tanh(B*(eng1-eng2))
        confusion_matrix[i][np.argmax(c)] += 1
print("testing error",(1 - np.sum(np.diagonal(confusion_matrix))/(Nc*100))*100)

