[![Build Status](https://travis-ci.com/chapman-phys220-2018f/CHANGEME.svg?branch=master)](https://travis-ci.com/chapman-phys220-2018f/CHANGEME)

# PHYS220/MATH220/CPSC220 CW 12

**Author(s):** **Enea Dodi & Jack Savage**

## Specification

Consider a ball of mass $m$ with horizontal coordinate $x$ rolling in a double-well potential $V(x) = x^4/4 - x^2/2$. (This is sometimes called the "sombrero" potential. Plot it to see why, for $x\in[-1.5,1.5]$.) This potential produces a force $f_{\text{hat}}(x) = -V'(x) = -x^3 + x$ on the rolling ball. Suppose the ball also has slight friction, so experiences a drag force $f_{\text{drag}}(\dot{x}) = -\nu \dot{x}$. With these forces we thus expect the ball to roll down the sides of the sombrero potential and settle in one of the two stable wells. However, this is boring, so instead we are going to shake the hat back and force periodically with a driving force $f_{\text{drive}}(t) = F\cos(\omega t)$. For small driving forces $F$, this should simply jiggle the ball back and forth at the bottom of one of the stable wells. Our task will be to explore what happens for larger driving forces $F$.

Note that according to Newton's second law, the ball must satisfy the equation of motion: $$m\ddot{x} = f_{\text{hat}}(x) + f_{\text{drag}}(\dot{x}) + f_{\text{drive}}(t) = x - x^3 - \nu \dot{x} + F\cos(\omega t)$$ 
This system is known as a periodically driven nonlinear "Duffing oscillator," and can be split into a set of two coupled first-order ODEs:
$$\dot{x}(t) = y(t)$$
$$m\dot{y}(t) = -\nu y(t) + x(t) - x^3(t) + F\cos(\omega t)$$
Your task will be to solve these equations numerically, for $m=1$, $\nu = 0.25$, and $\omega = 1$. Use a time-step size of $\Delta t = 0.001$ with the 4th-order Runge-Kutta integration method to keep sufficient numerical precision. Implement your code in a python file ```sombrero.py``` with suitable test functions in ```test_sombrero.py```.

Write a report in a notebook ```cw12-duffing.ipynb``` that summarizes the following investigations.

1. For $F = 0.18$, plot $x(t)$ for $t\in[0,2\pi\, 50]$, with $x(0) = -0.9$ and $y(0) = 0$. Plot the parametric curve $(x(t),y(t))$ in the x-y plane for the same time range.  Plot a scatter plot of the points $(x(t),y(t))$, with specific $t = n 2\pi$ for $n = 0,1,\ldots,50$. (This last type of plot is known as a "Poincare section" of the parametric curve.)  Interpret your results.
1. Repeat the first question, but for $x(0) = 0.9$ and $y(0) = 0$.  What is different?  Interpret your results.
1. Repeat the first question, but for $F = 0.25$, $x(0) = 0.2$, and $y(0) = 0.1$.  What is different?  Plot what happens if you change to $x(0) = 0.201$. How do you interpret this physically?
1. Repeat the first question, but for $F = 0.4$, $x(0) = 0$, and $y(0) = 0$.  What is different now?  What happens when you tweak the initial conditions?  How do you interpret this physically?
1. For the previous problem, extend the time range to $t\in[0,2\pi\, 1000]$.  What is the structure of the Poincare section?

Your homework this week will be to finish this classwork assignment, and then use it to create high-resolution plots of the Poincare section.

## Assessment

Analyze in this section what you found useful about this assignment in your own words. Include any lingering questions or comments that you may have. You can use the GitHub web interface to edit this file directly for now.

**Because of the similarity with the final, I decided to attempt to solve the problem similarily to how you solved the final, cutting down the processing time significantly. My implementation didn't work out, so I resort sadly to the old method of creating the dataframe. PS, if there is any relationship between the graphs and large interstellar bodies, I would like to know**

## Honor Pledge

I pledge that all the work in this repository is my own with only the following exceptions:

* Content of starter files supplied by the instructor;
* Code borrowed from another source, documented with correct attribution in the code and summarized here.

Signed,

**Enea Dodi**
