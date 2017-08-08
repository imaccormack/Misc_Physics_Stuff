#!/usr/bin/env python

import random
import Tkinter
import numpy as numpy
import scipy
import math
from numpy import random
from numpy import linalg
from scipy import integrate
from scipy import linalg
import pylab as P
import matplotlib.animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#from showmat import showmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

def E(s, params):
    L= len(s) #s is the spin configuration in region B
    tot= 0.0
    ep= params[0]
    J= params[1]
    for i in range(L):
        temp1= ep*s[i]
        temp2= J*s[i]*s[(i+1)%L]
        tot+= temp1 + temp2
    return tot #+ J*(tot**3)

def randState(L, fixedM= False, M=0):
    if not fixedM:
        return 2.0*(numpy.round(numpy.random.random(L)) - 0.5)
    else:
        Nup= (L+M)/2
        Ndown= (L-M)/2
        ups= numpy.ones(Nup)
        downs= -numpy.ones(Ndown)
        result= numpy.concatenate((ups,downs))
        numpy.random.shuffle(result)
        return result
    
def randStep(s,L):
    nFlips= numpy.random.poisson(10)
    bitsToFlip= numpy.random.randint(0,L,nFlips)
    result= s[:]
    for i in bitsToFlip:
        result[i]= -result[i]
    return result    

ZvsN= []
totvsN= []
totbyZ= []

sumvsF= []
Fs= []

def MCSum(f,N,L,params, fixedM,M):
    tot= 0.0
    #totvsN= []
    weight= N/(2.0**(L-1))
    Z= 0.0
    curr= randState(L)
    for i in range(N):
        x= f(curr, params) #Compute the weight for a given configuration using a given weight function, f
        negx= f(-curr, params)
        epx= math.exp(x)
        emx= math.exp(negx)
        rho= epx + emx
        if i>100: #This allows for a finite correlation time at the beginning of the random walk
            term1= (epx/rho)*(math.log(epx/(epx+emx)))
            term2= (emx/rho)*(math.log(emx/(epx+emx)))
            Z+= rho
            PdotS= -term1-term2
            tot+= PdotS
            sumvsF.append(PdotS)
            Fs.append(x)
            totvsN.append(tot)
            ZvsN.append(Z)
            totbyZ.append(tot/Z)
        willStep= False
        numFlips= 0
        while not willStep:
            tempConfig= randStep(curr,L)
            y= f(tempConfig, params) #Compute the weight for a given configuration using a given weight function, f
            negy= f(-tempConfig, params)
            epy= math.exp(y)
            emy= math.exp(negy)
            rhoy= epy + emy
            numFlips+=1
            if numpy.random.rand()<(rhoy/rho):
                curr= tempConfig
                willStep=True
            elif numFlips>50:
                curr= randState(L)
                willStep=True
        #tempweight= (i+1)/(2.0**(L-1))
        #totvsN.append(tot/tempweight)
    #Z= Z/(weight)
    #result= numpy.array(totvsN)
    #return result/Z
    return tot/(9899)#/Z
    
    
Ntest= 10000
eptest= 0.3
Ltest= 30
Jtests= numpy.linspace(-5.0,0.0,21)
#Jtests= [0.5]
possibleJs= []
sumvsJ= []
for j in Jtests:
    tempParams= [eptest,j]
    print "J: " + str(j)
    temp= MCSum(E,Ntest,Ltest,tempParams, False, 0.0)
    print "Result: " + str(temp)
    if temp > 0.005:
        possibleJs.append(j)
    sumvsJ.append(temp)
    
print possibleJs

plt.plot(Jtests,sumvsJ)
#plt.ylim([-0.5,1.0])
#plt.plot(totvsN)
#plt.plot(ZvsN, c="g")
#plt.plot(totbyZ)
#plt.plot(Ms, sumvsM)
#plt.scatter(Fs,sumvsF)
#hist,edges=numpy.histogram(Fs, bins=30)
#plt.scatter(edges[:-1],hist)
plt.show()

