#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Enea Dodi
# Student ID: 2296306
# Email: dodi@chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: Cw12
###

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numba as nb
from scipy.signal import argrelextrema
from mpl_toolkits.mplot3d import Axes3D




def dxdt(y):
    '''dxdt function
Args: y - the value at y(t)
Returns: x'(t)'''
    return np.float64(y)

@nb.jit
def solve_odes(xVal,yVal,F,m=1,v=0.25,w=1,dt=0.001, T = 100*np.pi):
    t = np.arange(0,T,dt)
    x = np.zeros_like(t, dtype= np.float64)
    y = np.zeros_like(t, dtype= np.float64)
    x[0] = np.float64(xVal)
    y[0] = np.float64(yVal)
    #rt = np.array(x[0],y[0])
    for i in range(1,len(t)):
        #rt = FourthOrderRungeKuttaR(dt,rt[0],rt[1],v,w,F,t)
        x[i] = FourthOrderRungeKuttaX(dt,y[i-1],x[i-1])
        y[i] = FourthOrderRungeKuttaY(dt,x[i-1],v,y[i-1],w,F,t[i-1])
        #x[i] = rt[0]
        #y[i] = rt[1]
    returnDataframe = pd.DataFrame({"t":t,"x":x,"y":y})
    return returnDataframe

@nb.jit
def drdt(x,v,y,w,F,t):
    return np.array([np.float64(y),(np.float64((-v*y)-x-(x**3)+F*np.cos(wt)))])

'''
def FourthOrderRungeKuttaR(dt,x,y,v,w,F,t):
    K1r = dt*(drdt(x,v,y,w,F,t))
    K2r = dt*((drdt(b,z,x,c)+np.array([K1r[0]/2,K1r[1]/2])).sum())
    K3r = dt*((drdt(x,v,y,w,F,t)+np.array([K2r[0]/2,K2r[1]/2])).sum())
    K4r = dt*((drdt(x,v,y,w,F,t)+np.array([K3r[0],K3r[1]])).sum()
    valx = x +(K1r[0]+2*K2r[0]+2*K3r[0]+K4r[0])/6
    valy = y + (K1r[1] + 2*K2r[1] + 2*K3r[1] + K4r[1])/6
    return np.array([valx,valy])
'''

@nb.jit
def FourthOrderRungeKuttaX(dt,y,x0):
    K1x = dt*(dxdt(y))
    K2x = dt*(dxdt(y)+K1x/2)
    K3x = dt*(dxdt(y)+K2x/2)
    K4x = dt*(dxdt(y)+K3x)
    return x0 + (K1x+2*K2x+2*K3x+K4x)/6

@nb.jit
def dydt(x,v,y,w,F,t):
    '''dydt function
    Args: x - the value at x(t)
      a - constant.
      y - the value at y(t)
    Returns: y'(t)'''
    return np.float((-v*y)-x-(x**3)+F*np.cos(w*t))

@nb.jit
def FourthOrderRungeKuttaY(dt,x,v,y,w,F,t):
    K1y = dt*(dydt(x,v,y,w,F,t))
    K2y = dt*(dydt(x,v,y,w,F,t)+K1y/2)
    K3y = dt*(dydt(x,v,y,w,F,t)+K2y/2)
    K4y = dt*(dydt(x,v,y,w,F,t)+K3y)
    return y + (K1y+2*K2y+2*K3y+K4y)/6
              
@nb.jit
def ploty(sol):
    '''ploty function
Args: sol - the panda dataframe
Returns: Plots y v t'''
    s = plt.figure(figsize=(8,6))
    a = plt.axes()
    a.plot(sol["t"],sol["y"], color = "red")
    a.set_xlabel("t values")
    a.set_ylabel("y values")
    a.set_title("y vs t")
    plt.show()

@nb.jit
def plotx(sol):
    '''plotx function
Args: sol - the panda dataframe
Returns: Plots x v t'''
    s = plt.figure(figsize=(8,6))
    a = plt.axes()
    a.plot(sol["t"],sol["x"], color = "blue")
    a.set_xlabel("t values")
    a.set_ylabel("x values")
    a.set_title("x vs t")
    plt.show()
    
@nb.jit
def plotxy(sol):
    '''plotxy function
Args: sol - the panda dataframe
Returns: Plots x v y'''
    s = plt.figure(figsize=(8,6))
    a = plt.axes()
    xvals = sol["x"]
    yvals = sol["y"]
    a.plot(xvals,yvals, color = "blue")
    a.set_xlabel("x values")
    a.set_ylabel("y values")
    a.set_title("y vs x")
    plt.show()

@nb.jit
def scatter(xVal,yVal,F,dt=0.001, N=50):
    TVal = 2*np.pi
    fig = plt.figure()
    for i in range(1,50):
        odes = solve_odes(xVal,yVal,F,T = i*TVal)
        if i == 0:
            plt.scatter(xVal,yVal,marker=".", color = 'k')
        else:
            index = (int)(i/dt)
            plt.scatter(odes["x"][:index],odes["y"][:index],marker ='.', color = 'k')
    plt.show()