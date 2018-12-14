#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###
# Name: Enea Dodi
# Student ID: 2296306
# Email: dodi@chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2018
# Assignment: Cw12
###
'''Sombrero module
Module containg all the code for Cw12'''
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
    '''solve_odes method
    args: xVal - inital x value
          yVal - inital y value
          F = driving force
    returns: a pd.DataFrame containing np.arrays of
    the points calcualated by FourthOrderRungeKutta.'''
    t = np.arange(0,T,dt)
    x = np.zeros_like(t, dtype= np.float64)
    y = np.zeros_like(t, dtype= np.float64)
    x[0] = np.float64(xVal)
    y[0] = np.float64(yVal)
    #rt = np.array(x[0],y[0])
    for i in range(1,len(t)):
        #rt = FourthOrderRungeKuttaR(dt,rt,v,w,F,t,m)
        x[i] = FourthOrderRungeKuttaX(dt,y[i-1],x[i-1])
        y[i] = FourthOrderRungeKuttaY(dt,x[i-1],v,y[i-1],w,F,t[i-1])
        #x[i] = rt[0]
        #y[i] = rt[1]
    returnDataframe = pd.DataFrame({"t":t,"x":x,"y":y})
    return returnDataframe

@nb.jit
def drdt(x,v,y,w,F,t,m):
    '''drdt method
    My attempt on creating a method which calculates both derivatives on the same time.'''
    return np.array([np.float64(y),dydt(x,v,y,w,F,t)/m])

@nb.jit
def FourthOrderRungeKuttaR(dt,r,v,w,F,t,m):
    '''FourthOrderRungeKuttaXs
    calculates the next x,y points using the RungeKuttaMethod'''
    K1r = dt*(drdt(r[0],v,r[1],w,F,t,m))
    K2r = dt*(drdt(r[0]+K1r[0]/2,v,r[1]+K1r[1]/2,w,F,(t+dt/2),m))
    K3r = dt*(drdt(r[0]+K2r[0]/2,v,r[1]+K2r[1]/2,w,F,(t+dt/2),m))
    K4r = dt*(drdt(r[0]+K1r[0],v,r[1]+K1r[1],w,F,(t+dt),m))
    return (r + (K1r+2*K2r+2*K3r+K4)/6)


@nb.jit
def FourthOrderRungeKuttaX(dt,y,x0):
    '''FourthOrderRungeKuttaXs
    calculates the next x point using the RungeKuttaMethod'''
    K1x = dt*(dxdt(y))
    K2x = dt*(dxdt(y)+K1x/2)
    K3x = dt*(dxdt(y)+K2x/2)
    K4x = dt*(dxdt(y)+K3x)
    return x0 + (K1x+2*K2x+2*K3x+K4x)/6

@nb.jit
def dydt(x,v,y,w,F,t):
    '''dydt function
    Args: x,v,y,w,F,t
    Returns: y'(t)'''
    return np.float((-v*y)+x-(x**3)+F*np.cos(w*t))

@nb.jit
def FourthOrderRungeKuttaY(dt,x,v,y,w,F,t):
    '''FourthOrderRungeKuttaY
    calculates the next y point using the RungeKuttaMethod'''
    K1y = dt*(dydt(x,v,y,w,F,t))
    K2y = dt*(dydt(x,v,y,w,F,t+dt/2)+K1y/2)
    K3y = dt*(dydt(x,v,y,w,F,t+dt/2)+K2y/2)
    K4y = dt*(dydt(x,v,y,w,F,t+dt)+K3y)
    return y + (K1y+2*K2y+2*K3y+K4y)/6
              
@nb.jit
def ploty(sol,title):
    '''ploty function
Args: sol - the panda dataframe
Returns: Plots y v t'''
    s = plt.figure(figsize=(8,6))
    a = plt.axes()
    a.plot(sol["t"],sol["y"], color = "red")
    a.set_xlabel("t values")
    a.set_ylabel("y values")
    a.set_title(title)
    plt.show()

@nb.jit
def plotx(sol,title):
    '''plotx function
Args: sol - the panda dataframe
Returns: Plots x v t'''
    s = plt.figure(figsize=(8,6))
    a = plt.axes()
    a.plot(sol["t"],sol["x"], color = "blue")
    a.set_xlabel("t values")
    a.set_ylabel("x values")
    a.set_title(title)
    plt.show()
    
@nb.jit
def plotxy(sol,title):
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
    a.set_title(title)
    plt.show()

@nb.jit
def scatter(odes,F,title,dt=0.001, N=50):
    '''scatter method
    Args: odes - the pd.Dataframe containg the calculated results from solve_odes
          title - title of the graph
          F - driving Force
    returns: a Poincare section'''
    tVal = 2*np.pi
    fig = plt.figure()
    for i in range(0,N+1):
        plt.scatter(odes["x"][(int)(tVal*i)],odes["y"][(int)(tVal*i)],marker='.',color = 'k')
    plt.title(title)
    plt.show()