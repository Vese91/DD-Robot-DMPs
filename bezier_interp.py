'''
Given a set of points, we want to interpolate them using cubic Bezier curves. 
The goal is to fit n+1 given points (P0,...,Pn). In order to fit these points,
we are going to use one cubic Bézier curve (4 control points) between each 
consecutive pair of points. Use "evaluate_bezier" function to evaluate the
interpolated points.

Inputs: 
- points: list of points to interpolate
- n: number of points to evaluate each cubic curve

Outputs:
- bezier_interp: interpolated points
'''

import numpy as np

def get_bezier_coeff(points):
  '''
  Cubic Bezier interpolation coefficients

  Input:
  - points: list of points to interpolate

  Output:
  - A, B: coefficients of the cubic Bezier curve
  '''
  n = len(points)-1

  # Build coefficients matrix
  C = 4*np.identity(n)
  np.fill_diagonal(C[1:],1)
  np.fill_diagonal(C[:,1:],1)
  C[0,0] = 2
  C[n-1,n-1] = 7
  C[n-1,n-2] = 2

  # Build points vector
  P = [2*(2*points[i]+points[i+1]) for i in range(n)]
  P[0] = points[0]+2*points[1]
  P[n-1] = 8*points[n-1]+points[n]

  # Solve system, find A and B
  A = np.linalg.solve(C,P)
  B = [0]*n
  for i in range(n - 1):
      B[i] = 2*points[i+1]-A[i+1]

  B[n-1] = (A[n-1]+points[n])/2

  return A, B

def get_cubic(a,b,c,d):
    '''
    Cubic Bezier curve function. 
    A cubic Bezier curve is defined by 4 control points.
    
    Inputs:
    - a: initial point
    - b: first control point
    - c: second control point
    - d: final point
    
    Output:
    - lambda function that represents the cubic Bezier curve
    '''
    # Returns the general Bezier cubic formula given 4 control points
    cb_expression = lambda t: np.power(1-t,3)*a+3*np.power(1-t,2)*t*b+3*(1-t)*np.power(t,2)*c+np.power(t,3)*d
    cb_velocity = lambda t: 3*np.power(1-t,2)*(b-a)+6*(1-t)*t*(c-b)+3*np.power(t,2)*(d-c)
    return cb_expression, cb_velocity
       
def get_bezier_cubic(points):
    '''
    Cubic Bezier interpolation.
    
    Inputs:
    - points: list of points to interpolate
    
    Output:
    - list of cubic Bezier curves that interpolate the given points
    '''
    # Return one cubic curve for each consecutive points
    A, B = get_bezier_coeff(points)
    curves = []
    vels = []
    for i in range(len(points)-1):
        curve, vel = get_cubic(points[i], A[i], B[i], points[i+1])
        curves.append(curve)
        vels.append(vel)
    return curves, vels

def evaluate_bezier(points, n):
    '''
    Evaluate cubic Bezier interpolation given a set of points.
    
    Inputs:
    - points: list of points to interpolate
    - n: number of points to evaluate each cubic curve

    Outputs:
    - bezier_interp: interpolated points
    '''
    curves, vels = get_bezier_cubic(points)
    timeVec = np.linspace(0,1,n)
    bezier_interp = []
    bezier_vel = []
    for fun in curves:
        for t in timeVec:
            segment = fun(t)
            bezier_interp.append(segment)

    for fun in vels:
        for t in timeVec:
            segment = fun(t)
            bezier_vel.append(segment)
    
    bezier_interp = np.array(bezier_interp)
    bezier_vel = np.array(bezier_vel)

    return bezier_interp, bezier_vel

  
    