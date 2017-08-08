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
        temp2= J*s[i]*s[(i+1)%L]*s[(i+2)%L]
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

def ZPiSi(f, params, config):
    x= f(config, params) #Compute the weight for a given configuration using a given weight function, f
    negx= f(-config, params)
    epx= math.exp(x)
    emx= math.exp(negx)
    term1= epx*math.log(epx/(epx+emx))
    term2= emx*math.log(emx/(epx+emx))
    return -term1-term2

ZvsN= []
totvsN= []
totbyZ= []

def MCSum(f,N,L,params, fixedM,M):
    tot= 0.0
    #totvsN= []
    weight= N/(2.0**(L-1))
    Z= 0.0
    for i in range(N):
        s= randState(L, fixedM, M)
        x= f(s, params) #Compute the weight for a given configuration using a given weight function, f
        negx= f(-s, params)
        epx= math.exp(x)
        emx= math.exp(negx)
        term1= epx*math.log(epx/(epx+emx))
        term2= emx*math.log(emx/(epx+emx))
        #term1= epx*(x - math.log(epx+emx))
        #term2= -emx*(x + math.log(epx+emx))
        Z+= epx + emx
        PdotS= -term1-term2
        tot+= PdotS
        totvsN.append(tot)
        ZvsN.append(Z)
        totbyZ.append(tot/Z)
        #tempweight= (i+1)/(2.0**(L-1))
        #totvsN.append(tot/tempweight)
    #Z= Z/(weight)
    #result= numpy.array(totvsN)
    #return result/Z
    return tot/Z
    
    
Ntest= 1000
eptest= 0.0
Ltest= 1000
#Jtests= numpy.linspace(-3.0,3.0,31)
Jtests= [0.6]
possibleJs= []
sumvsJ= []
sumvsM= []
Ms= []
for j in Jtests:
    tempParams= [eptest,j]
    print "J: " + str(j)
    temp= MCSum(E,Ntest,Ltest,tempParams, False, 0.0)
    """for m in range(-Ltest/2,Ltest/2 +1):
        nup= (m+Ltest)/2
        temp= MCSum(E,Ntest,Ltest,tempParams, True, m)
        #temp= temp*(scipy.misc.comb(Ltest,nup)/(2**Ltest))
        Ms.append(m)
        sumvsM.append(temp)"""
    print "Result: " + str(temp)
    if temp > 0.005:
        possibleJs.append(j)
    sumvsJ.append(temp)
    
print possibleJs

#plt.plot(Jtests,sumvsJ)
#plt.ylim([-0.5,1.0])
plt.plot(totvsN)
#plt.plot(ZvsN, c="g")
#plt.plot(totbyZ)
#plt.plot(Ms, sumvsM)
plt.show()

