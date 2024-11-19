import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc
from dmp import dmp
import bezier_interp as bz
from cbf import CBF

# Dynamic parameters
mu_s = 0.1  # static friction coefficient
g = 9.81 # gravity acceleration [m/s^2]
alpha = 1 # extended class-K function parameter (straight line)
exp = 1 # exponent of the extended class-K function, it must be an odd number (leave it as 1)

 # Reference trajectory (Cartesian coordinates)
m = 15  # number of points between control points
waypoints = np.array([[1.5,3],[3.5,2.92],[5,2.7],[6,2],[5,1.31],[3.49,1.11],[1.5,1.0]]) # control points 
ref_path, ref_vel = bz.evaluate_bezier(waypoints, m)  # evaluate Interpolating Bezier curves

# Reference trajectory (Polar coordinates)
ref_path_polar, ref_vel_polar = bz.convert_to_polar_coord(ref_path, ref_vel) # convert Cartesian to Polar coordinates

# Comparison between Cartesian and Polar coordinates
plt.plot(ref_path[:,0], ref_path[:,1],'b-', label='Reference trajectory cartesian')
plt.plot(ref_path_polar[:,0]*np.cos(ref_path_polar[:,1]), ref_path_polar[:,0]*np.sin(ref_path_polar[:,1]),'r--', label='Reference trajectory polar')
plt.plot(waypoints[:,0], waypoints[:,1], 'ro', label='Control points')
plt.legend()
plt.show()

# Polar velocity
plt.plot(ref_vel_polar[:,0], label = 'rho_dot')
plt.plot(ref_vel_polar[:,1], label = 'theta_dot')
plt.legend()
plt.show()

print(">> End of the script")