#!/usr/bin/env python

from random import randint
from random import random
import Tkinter
import numpy as numpy
import scipy
import math
import progressbar
from math import e
from numpy import random as nr
from numpy import linalg
from scipy import linalg
from math import factorial
import pylab as P
import matplotlib.animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
#from showmat import showmat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import *
from scipy.sparse.linalg import eigsh

class RBMstate(object):
    
    def __init__(self, Ham, Winit, ainit, binit):
        self.a= ainit
        self.b= binit
        self.W= Winit
        self.Nvis= len(ainit)
        self.Nhid= len(binit)
        self.alpha= self.Nhid/self.Nvis
        self.H= Ham
        self.currE= self.Avg2()
        self.convTest= []
    
    def Psi(self, S):
        temp= numpy.dot(self.a,S)
        F= numpy.cosh((self.b + numpy.dot(S,self.W)))
        """print "The first term in Psi is"
        print """
        return numpy.exp(temp)*numpy.prod(F)
    
    def randState(self):
        result= 2.0*(numpy.round(nr.random(self.Nvis)) - 0.5)
        return result
    
    def flipSpin(self, s):
        numflips= 1#randint(0,5)
        result= s[:]
        for i in range(numflips):
            ind= randint(0,self.Nvis-1)
            result[ind]= -result[ind]
        return result
    
    def Avg(self):
        corr= self.Nvis
        N= self.Nvis**2 + corr
        norm= 0.0
        s= self.randState()
        psis= self.Psi(s)
        resultOp= numpy.zeros(self.Nvis+self.Nhid+self.Nvis*self.Nhid)
        resultOpOp= numpy.zeros((self.Nvis+self.Nhid+self.Nvis*self.Nhid,self.Nvis+self.Nhid+self.Nvis*self.Nhid))
        resultE= 0.0
        resultEOp= resultOp
        for i in range(N):
            if i>corr:
                weight= numpy.absolute(psis)**2
                norm+= weight
                ops= self.O(s)
                Es= self.H(s)
                resultOp+= weight*ops
                resultOpOp+= weight*numpy.outer(numpy.conj(ops), ops)
                resultE+= weight*Es
                resultEOp+= weight*Es*numpy.conj(ops)
            #s= self.randState()
            change= False
            n=0
            while not change:
                temp= self.flipSpin(s)
                #tempWeight= numpy.absolute(self.Psi(temp))**2
                p= numpy.absolute(self.Psi(temp)/psis)**2 #tempWeight/weight
                change= (nr.rand(1)[0] < p)
                n+=1
                #if n>50:
                    #change= True
                if change:
                    if n>50:
                        s= self.randState()
                    else:
                        s= temp
                        psis= self.Psi(s)
        return [resultOp/norm, resultOpOp/norm, resultE/norm, resultEOp/norm]
    
    def O(self, s):
        result= numpy.zeros(self.Nvis+self.Nhid+self.Nvis*self.Nhid)
        result[0:self.Nvis]= s[0:self.Nvis]
        theta= self.b + numpy.dot(s,self.W)
        result[self.Nvis:self.Nvis+self.Nhid]= numpy.tanh(theta)
        for j in range(self.Nvis):
            result[self.Nvis+self.Nhid + j*self.Nhid:self.Nvis+self.Nhid+(j+1)*self.Nhid]= s[j]*numpy.tanh(theta)
        return result
    
    def dW(self,n):
        averages= self.Avg()
        avgO= averages[0]
        Smat= averages[1] - numpy.outer(numpy.conj(avgO), avgO)
        ep= 0.001#max([100*(0.9**n),0.0001])
        Smat= Smat + ep*numpy.diag(numpy.diag(Smat))#ep*identity(self.Nvis+self.Nhid+self.Nvis*self.Nhid)
        avgEO= averages[3]
        self.currE= averages[2]
        F= avgEO - self.currE*numpy.conj(avgO)
        invSmat= numpy.linalg.inv(Smat)#pinv(Smat)
        return numpy.dot(invSmat,F)
    
    def update(self,n):
        deltaW= self.dW(n)
        deltaT= 0.25#(0.99**n)
        self.a= self.a - deltaT*deltaW[0:self.Nvis]
        self.b= self.b - deltaT*deltaW[self.Nvis:self.Nvis+self.Nhid]
        for j in range(self.Nvis):
            self.W[j]= self.W[j] - deltaT*deltaW[self.Nvis+self.Nhid + j*self.Nhid:self.Nvis+self.Nhid+(j+1)*self.Nhid]
        self.convTest.append(self.currE)
        
    def Avg2(self):
        corr= self.Nvis
        N= self.Nvis**2 + corr
        norm= 0.0
        s= self.randState()
        psis= self.Psi(s)
        resultE= 0.0
        for i in range(N):
            if i>corr:
                weight= numpy.absolute(psis)**2
                norm+= weight
                Es= self.H(s)
                resultE+= weight*Es
            #s= self.randState()
            change= False
            n=0
            while not change:
                temp= self.flipSpin(s)
                #tempWeight= numpy.absolute(self.Psi(temp))**2
                p= numpy.absolute(self.Psi(temp)/psis)**2 #tempWeight/weight
                change= (nr.rand(1)[0] < p)
                n+=1
                #if n>50:
                    #change= True
                if change:
                    if n>50:
                        s= self.randState()
                    else:
                        s= temp
                        psis= self.Psi(s)
        return resultE/norm
        
    def train2(self, Ntrain):
        pbar=progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',])
        for i in pbar(range(Ntrain)):
            self.a= 2.0*(nr.random(self.Nvis)*2.0 -1.0)
            self.b= 1.0*(nr.random(self.Nhid)*2.0 -1.0)
            self.W= 1.0*(nr.random((self.Nvis,self.Nhid))*2.0 -1.0)
            temp= self.Avg2()
            if temp<self.currE:
                self.currE= temp
            self.convTest.append(self.currE)
    
            
    def train(self, Ntrain):
        pbar=progressbar.ProgressBar(widgets=[
        ' [', progressbar.Timer(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',])
        for i in pbar(range(Ntrain)):
            if i%100==0:
                print self.b
                print max(numpy.absolute(self.a))
                print max(numpy.absolute(self.b))
                print max(numpy.absolute(numpy.ndarray.flatten(self.W)))
            self.update(i)
            
    
    
    
N= 5
M= 5
aTest= 10.0*(nr.random(N)*2.0 -1.0)
bTest= 10.0*(nr.random(M)*2.0 -1.0)
wTest= 10.0*(nr.random((N,M))*2.0 -1.0)
Ntrain= 5000

def isingH(s):
    J= -0.2
    result= 0.0
    for i in range(len(s)):
        result+= s[i]*s[(i+1)%len(s)]
    return result*J

def LFisingH(s):
    J=-0.02
    for i in range(len(s)):
        result+= s[i]*s[(i+1)%len(s)]
    return result*J

theState= RBMstate(isingH,wTest,aTest,bTest)
theState.train2(Ntrain)

print theState.convTest[-1]
"""newState= RBMstate(isingH, theState.W, theState.a, theState.b)
print newState.currE"""
plt.plot(range(Ntrain),theState.convTest)
plt.ylim([-3.0,3.0])
plt.show()

"""Store monte carlo states to speed things up"""
    
    
    